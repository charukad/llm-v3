"""
Health check endpoints for the API.
Part of the Production Readiness implementation (Sprint 21).
"""
import time
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import psutil
import os
import platform
from datetime import datetime, timedelta

from math_llm_system.orchestration.agents.registry import AgentRegistry
from math_llm_system.orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus
from math_llm_system.database.access.mongodb_wrapper import MongoDBWrapper
from math_llm_system.core.mistral.inference import MistralInference

router = APIRouter(prefix="/health", tags=["health"])

# System start time for uptime calculation
START_TIME = datetime.now()

# Cache for expensive health checks
health_cache = {
    "last_check": {},
    "results": {}
}
CACHE_TTL = 30  # seconds

def get_agent_registry():
    """Dependency for agent registry."""
    return AgentRegistry()

def get_message_bus():
    """Dependency for message bus."""
    return RabbitMQBus()

def get_mongodb():
    """Dependency for MongoDB wrapper."""
    return MongoDBWrapper()

def get_model_inference():
    """Dependency for model inference."""
    return MistralInference.get_instance()

def format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{int(days)}d")
    if hours > 0 or days > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{int(seconds)}s")
    
    return " ".join(parts)

def get_cached_or_compute(cache_key: str, compute_func, ttl: int = CACHE_TTL):
    """Get a cached value or compute and cache it if expired or missing."""
    now = time.time()
    last_check = health_cache["last_check"].get(cache_key, 0)
    
    # If cache is valid, return cached result
    if now - last_check < ttl and cache_key in health_cache["results"]:
        return health_cache["results"][cache_key]
    
    # Compute new result
    result = compute_func()
    
    # Update cache
    health_cache["last_check"][cache_key] = now
    health_cache["results"][cache_key] = result
    
    return result

@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """
    General health check endpoint.
    Returns basic system health information.
    """
    uptime_seconds = (datetime.now() - START_TIME).total_seconds()
    
    # Get system resource usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "status": "healthy",
        "version": "0.85.0",  # Based on 85% completion from status report
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "uptime": format_uptime(uptime_seconds),
        "uptime_seconds": uptime_seconds,
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/database", response_model=Dict[str, Any])
async def database_health(mongodb: MongoDBWrapper = Depends(get_mongodb)):
    """
    Check database health.
    Tests connection and provides basic statistics.
    """
    def check_database():
        try:
            # Measure ping time
            start = time.time()
            client = mongodb.get_client()
            client.admin.command('ping')
            ping_time = (time.time() - start) * 1000  # Convert to ms
            
            # Get basic stats if possible
            db_stats = {}
            try:
                db = client.get_database()
                db_stats = {
                    "collections": len(db.list_collection_names()),
                    "database_name": db.name
                }
            except Exception:
                # If detailed stats fail, we still have basic connectivity
                pass
                
            return {
                "status": "connected",
                "ping_time_ms": round(ping_time, 2),
                **db_stats
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    result = get_cached_or_compute("database_health", check_database)
    
    if result["status"] != "connected":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database is not available: {result.get('error')}"
        )
    
    return result

@router.get("/message-bus", response_model=Dict[str, Any])
async def message_bus_health(message_bus: RabbitMQBus = Depends(get_message_bus)):
    """
    Check message bus health.
    Tests connection and provides queue information.
    """
    def check_message_bus():
        try:
            # Check connection
            if not message_bus.is_connected():
                return {
                    "status": "disconnected",
                    "error": "Not connected to RabbitMQ"
                }
            
            # Get queue information
            queues = message_bus.list_queues()
            
            return {
                "status": "connected",
                "queues": queues,
                "queue_count": len(queues)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    result = get_cached_or_compute("message_bus_health", check_message_bus)
    
    if result["status"] != "connected":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Message bus is not available: {result.get('error')}"
        )
    
    return result

@router.get("/agents", response_model=Dict[str, Any])
async def agents_health(agent_registry: AgentRegistry = Depends(get_agent_registry)):
    """
    Check health of all agents.
    Returns availability and basic status of each agent.
    """
    def check_agents():
        try:
            # Get all registered agents
            agents = agent_registry.get_all_agents()
            
            if not agents:
                return {
                    "status": "warning",
                    "message": "No agents registered",
                    "count": 0,
                    "agents": {}
                }
            
            # Check health of each agent
            agent_statuses = {}
            healthy_count = 0
            
            for agent in agents:
                agent_id = agent["id"]
                try:
                    # Try to ping the agent
                    is_healthy = agent_registry.ping_agent(agent_id)
                    
                    if is_healthy:
                        status = "healthy"
                        healthy_count += 1
                    else:
                        status = "unhealthy"
                        
                    agent_statuses[agent_id] = {
                        "status": status,
                        "capabilities": agent.get("capabilities", []),
                        "last_seen": agent.get("last_seen")
                    }
                except Exception as e:
                    agent_statuses[agent_id] = {
                        "status": "error",
                        "error": str(e),
                        "capabilities": agent.get("capabilities", [])
                    }
            
            # Determine overall status
            overall_status = "healthy"
            if healthy_count == 0:
                overall_status = "critical"
            elif healthy_count < len(agents):
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "count": len(agents),
                "healthy_count": healthy_count,
                "agents": agent_statuses
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agents": {}
            }
    
    result = get_cached_or_compute("agents_health", check_agents)
    
    if result["status"] == "critical" or result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent system is not available: {result.get('error', 'No healthy agents')}"
        )
    
    return result

@router.get("/model", response_model=Dict[str, Any])
async def model_health(model: MistralInference = Depends(get_model_inference)):
    """
    Check model health.
    Tests if the model is loaded and responsive.
    """
    def check_model():
        try:
            # Check if model is loaded
            if not model.is_loaded():
                return {
                    "status": "not_loaded",
                    "error": "Model is not loaded"
                }
            
            # Test simple inference
            start = time.time()
            test_output = model.generate("2+2=")
            inference_time = (time.time() - start) * 1000  # Convert to ms
            
            # Check if output is reasonable (contains "4")
            is_reasonable = "4" in test_output
            
            status = "loaded" if is_reasonable else "error"
            
            return {
                "status": status,
                "name": model.get_model_name(),
                "quantization": model.get_quantization_info(),
                "load_time_ms": model.get_load_time_ms(),
                "inference_time_ms": round(inference_time, 2),
                "test_passed": is_reasonable
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    result = get_cached_or_compute("model_health", check_model)
    
    if result["status"] not in ["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not available: {result.get('error', 'Unknown error')}"
        )
    
    return result

@router.get("/all", response_model=Dict[str, Any])
async def all_health_checks(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    message_bus: RabbitMQBus = Depends(get_message_bus),
    mongodb: MongoDBWrapper = Depends(get_mongodb),
    model: MistralInference = Depends(get_model_inference)
):
    """
    Run all health checks.
    Comprehensive system health status.
    """
    # Run all health checks
    base_health = await health_check()
    
    try:
        database_health_result = await database_health(mongodb)
    except HTTPException:
        database_health_result = {"status": "error"}
    
    try:
        message_bus_health_result = await message_bus_health(message_bus)
    except HTTPException:
        message_bus_health_result = {"status": "error"}
    
    try:
        agents_health_result = await agents_health(agent_registry)
    except HTTPException:
        agents_health_result = {"status": "error"}
    
    try:
        model_health_result = await model_health(model)
    except HTTPException:
        model_health_result = {"status": "error"}
    
    # Determine overall system health
    components = [
        database_health_result,
        message_bus_health_result,
        agents_health_result,
        model_health_result
    ]
    
    error_count = sum(1 for c in components if c["status"] in ["error", "critical"])
    warning_count = sum(1 for c in components if c["status"] in ["warning", "degraded"])
    
    if error_count > 0:
        overall_status = "critical"
    elif warning_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        **base_health,
        "overall_status": overall_status,
        "components": {
            "database": database_health_result,
            "message_bus": message_bus_health_result,
            "agents": agents_health_result,
            "model": model_health_result
        }
    }

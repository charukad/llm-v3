"""
Performance monitoring and optimization endpoints.
Provides API access to performance metrics, diagnostics, and optimization controls.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import time
import psutil
import os
import json

from math_llm_system.orchestration.performance.performance_optimizer import PerformanceOptimizer
from math_llm_system.orchestration.performance.resource_manager import ResourceManager
from math_llm_system.math_processing.computation.computation_cache import ComputationCache
from math_llm_system.database.optimization.query_optimizer import QueryOptimizer
from math_llm_system.database.access.mongodb_wrapper import MongoDBWrapper
from math_llm_system.orchestration.monitoring.metrics import MetricsCollector

router = APIRouter(prefix="/performance", tags=["performance"])

# Dependencies
def get_performance_optimizer():
    """Get performance optimizer singleton."""
    return PerformanceOptimizer()

def get_resource_manager():
    """Get resource manager singleton."""
    return ResourceManager()

def get_computation_cache():
    """Get computation cache singleton."""
    return ComputationCache()

def get_mongodb():
    """Get MongoDB wrapper instance."""
    return MongoDBWrapper()

def get_query_optimizer(mongodb: MongoDBWrapper = Depends(get_mongodb)):
    """Get query optimizer instance."""
    return QueryOptimizer(mongodb.get_client())

def get_metrics_collector():
    """Get metrics collector instance."""
    return MetricsCollector()

@router.get("/", response_model=Dict[str, Any])
async def get_performance_overview(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """
    Get overall performance metrics and system health.
    Provides a comprehensive overview of system performance.
    """
    # Get system resource usage
    resource_usage = resource_manager.get_resource_usage()
    
    # Get optimization metrics
    optimization_metrics = optimizer.get_metrics()
    
    # Get performance recommendations
    recommendations = optimizer.get_recommendations()
    
    return {
        "status": "healthy" if resource_usage["cpu"]["percent"] < 80 else "high_load",
        "timestamp": time.time(),
        "system_resources": resource_usage,
        "optimization_metrics": optimization_metrics,
        "recommendations": recommendations[:5] if recommendations else []
    }

@router.get("/detailed", response_model=Dict[str, Any])
async def get_detailed_performance_metrics(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    resource_manager: ResourceManager = Depends(get_resource_manager),
    computation_cache: ComputationCache = Depends(get_computation_cache),
    query_optimizer: QueryOptimizer = Depends(get_query_optimizer),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Get detailed performance metrics for all system components.
    Comprehensive performance diagnostics for the entire system.
    """
    # Get resource usage history
    resource_history = resource_manager.get_resource_history()
    
    # Get active tasks
    active_tasks = resource_manager.get_active_tasks()
    
    # Get cache metrics
    cache_metrics = computation_cache.get_metrics()
    
    # Get database optimization report
    db_optimization = query_optimizer.generate_optimization_report()
    
    # Get component-specific metrics
    component_metrics = metrics_collector.get_all_metrics()
    
    # Get request latency metrics
    latency_metrics = metrics_collector.get_latency_metrics()
    
    return {
        "timestamp": time.time(),
        "resource_usage": resource_manager.get_resource_usage(),
        "resource_history": {
            "cpu_usage": resource_history["cpu_usage"][-20:],  # Last 20 data points
            "memory_usage": resource_history["memory_usage"][-20:],
            "gpu_usage": resource_history["gpu_usage"][-5:] if resource_history["gpu_usage"] else []
        },
        "task_management": {
            "active_tasks": active_tasks,
            "active_count": len(active_tasks)
        },
        "cache_performance": cache_metrics,
        "database_optimization": {
            "slow_queries": db_optimization["slow_queries"][:5],
            "index_recommendations": db_optimization["index_recommendations"][:5],
            "unused_indexes": db_optimization["unused_indexes"]
        },
        "component_metrics": component_metrics,
        "latency_metrics": latency_metrics,
        "optimization_status": optimizer.get_status()
    }

@router.post("/optimize", response_model=Dict[str, Any])
async def run_performance_optimization(
    background_tasks: BackgroundTasks,
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    optimization_level: Optional[int] = None,
    components: Optional[List[str]] = None
):
    """
    Run system-wide performance optimization.
    Analyzes and optimizes various system components.
    
    Args:
        optimization_level: Level of optimization (1-3)
        components: Specific components to optimize (None for all)
    """
    # Start optimization in background
    background_tasks.add_task(
        optimizer.run_optimization,
        level=optimization_level,
        components=components
    )
    
    return {
        "status": "optimization_started",
        "message": "Performance optimization started in background",
        "optimization_level": optimization_level or optimizer.get_default_level(),
        "components": components or "all"
    }

@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_statistics(
    computation_cache: ComputationCache = Depends(get_computation_cache)
):
    """
    Get statistics about the computation cache.
    Provides metrics on cache hit rate, size, and effectiveness.
    """
    cache_metrics = computation_cache.get_metrics()
    cache_health = computation_cache.get_health()
    
    return {
        "metrics": cache_metrics,
        "health": cache_health,
        "status": "healthy" if cache_health["redis_available"] else "degraded"
    }

@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache(
    computation_cache: ComputationCache = Depends(get_computation_cache)
):
    """
    Clear the computation cache.
    Removes all cached computation results.
    """
    computation_cache.clear()
    
    return {
        "status": "success",
        "message": "Computation cache cleared",
        "timestamp": time.time()
    }

@router.get("/database/stats", response_model=Dict[str, Any])
async def get_database_statistics(
    query_optimizer: QueryOptimizer = Depends(get_query_optimizer),
    mongodb: MongoDBWrapper = Depends(get_mongodb)
):
    """
    Get database performance statistics.
    Provides metrics on query performance, indexes, and optimization.
    """
    # Get statistics for key collections
    collection_stats = {}
    client = mongodb.get_client()
    db = client[mongodb.get_database_name()]
    
    for collection_name in ["conversations", "interactions", "expressions", "visualizations"]:
        try:
            collection = db[collection_name]
            stats = query_optimizer.get_collection_statistics(collection)
            collection_stats[collection_name] = stats
        except Exception as e:
            collection_stats[collection_name] = {"error": str(e)}
    
    # Get optimization report
    optimization_report = query_optimizer.generate_optimization_report()
    
    return {
        "collection_stats": collection_stats,
        "slow_queries": optimization_report["slow_queries"][:10],
        "index_recommendations": optimization_report["index_recommendations"],
        "unused_indexes": optimization_report["unused_indexes"],
        "timestamp": time.time()
    }

@router.post("/database/optimize", response_model=Dict[str, Any])
async def optimize_database(
    background_tasks: BackgroundTasks,
    query_optimizer: QueryOptimizer = Depends(get_query_optimizer),
    mongodb: MongoDBWrapper = Depends(get_mongodb),
    collections: Optional[List[str]] = None,
    apply_indexes: bool = False
):
    """
    Optimize database performance.
    Analyzes queries and optionally applies recommended indexes.
    
    Args:
        collections: List of collections to optimize (None for all)
        apply_indexes: Whether to apply recommended indexes
    """
    # Function to run in background
    def run_db_optimization():
        client = mongodb.get_client()
        db = client[mongodb.get_database_name()]
        
        # Determine collections to optimize
        collection_names = collections or db.list_collection_names()
        
        # Generate optimization report
        report = query_optimizer.generate_optimization_report()
        
        # Apply indexes if requested
        applied_indexes = []
        if apply_indexes:
            for collection_name in collection_names:
                collection = db[collection_name]
                indexes = query_optimizer.apply_recommended_indexes(collection)
                if indexes:
                    applied_indexes.extend(indexes)
        
        return {
            "slow_queries_optimized": len(report["slow_queries"]),
            "indexes_applied": applied_indexes,
            "collections_processed": collection_names
        }
    
    # Start optimization in background
    background_tasks.add_task(run_db_optimization)
    
    return {
        "status": "optimization_started",
        "message": "Database optimization started in background",
        "collections": collections or "all",
        "apply_indexes": apply_indexes
    }

@router.get("/resource-usage", response_model=Dict[str, Any])
async def get_resource_usage(
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """
    Get current system resource usage.
    Provides detailed metrics on CPU, memory, and GPU utilization.
    """
    # Get basic resource usage
    usage = resource_manager.get_resource_usage()
    
    # Get process-specific stats
    process = psutil.Process(os.getpid())
    process_stats = {
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
        "memory_info": {
            "rss": process.memory_info().rss / (1024 ** 2),  # MB
            "vms": process.memory_info().vms / (1024 ** 2),  # MB
        },
        "threads": len(process.threads()),
        "open_files": len(process.open_files()),
        "connections": len(process.connections())
    }
    
    return {
        "system": usage,
        "process": process_stats,
        "timestamp": time.time()
    }

@router.get("/latency", response_model=Dict[str, Any])
async def get_latency_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
    endpoint: Optional[str] = None,
    timeframe: str = "1h"
):
    """
    Get API latency metrics.
    Provides request latency statistics for various endpoints.
    
    Args:
        endpoint: Specific endpoint to get metrics for (None for all)
        timeframe: Time frame for metrics (e.g., "1h", "1d")
    """
    # Parse timeframe
    seconds = 3600  # Default 1 hour
    if timeframe.endswith("m"):
        seconds = int(timeframe[:-1]) * 60
    elif timeframe.endswith("h"):
        seconds = int(timeframe[:-1]) * 3600
    elif timeframe.endswith("d"):
        seconds = int(timeframe[:-1]) * 86400
    
    # Get latency metrics
    latency_metrics = metrics_collector.get_latency_metrics(
        endpoint=endpoint,
        since=time.time() - seconds
    )
    
    return {
        "metrics": latency_metrics,
        "endpoint": endpoint or "all",
        "timeframe": timeframe,
        "timestamp": time.time()
    }

@router.post("/set-optimization-level", response_model=Dict[str, Any])
async def set_optimization_level(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    level: int = Query(..., ge=0, le=3)
):
    """
    Set the global optimization level.
    Controls how aggressively the system applies optimizations.
    
    Args:
        level: Optimization level (0-3)
    """
    # Set optimization level
    optimizer.set_optimization_level(level)
    
    return {
        "status": "success",
        "message": f"Optimization level set to {level}",
        "level": level,
        "settings": optimizer.get_settings_for_level(level)
    }

@router.post("/reconfigure-resource-manager", response_model=Dict[str, Any])
async def reconfigure_resource_manager(
    resource_manager: ResourceManager = Depends(get_resource_manager),
    config: Dict[str, Any] = None
):
    """
    Reconfigure the resource manager.
    Updates settings for CPU, memory, and GPU resource allocation.
    
    Args:
        config: New configuration settings
    """
    if not config:
        raise HTTPException(status_code=400, detail="Configuration required")
    
    # Update configuration
    success = resource_manager.update_config(config)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update configuration")
    
    return {
        "status": "success",
        "message": "Resource manager configuration updated",
        "timestamp": time.time()
    }

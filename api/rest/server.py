"""
FastAPI server for the Mathematical Multimodal LLM System.

This module sets up the FastAPI application with REST and WebSocket
endpoints for the system.
"""
import logging
import os
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.rest.routes import math, multimodal, chat_visualization
from api.websocket.server import websocket_router
from api.rest.routes.visualization import router as visualization_router
from api.rest.routes.workflow import router as workflow_router
from api.rest.routes.query_analysis import router as query_analysis_router
from api.rest.middlewares.error_handler import ErrorHandler
from orchestration.workflow.workflow_engine import WorkflowEngine
from orchestration.manager.orchestration_manager import OrchestrationManager, get_orchestration_manager
from core.agent.llm_agent import CoreLLMAgent, initialize_llm_agent
from orchestration.agents.registry import get_agent_registry, register_core_agents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Mathematical Multimodal LLM System",
    description="API for processing mathematical content across multiple modalities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add global error handling middleware
app.middleware("http")(ErrorHandler())

# Include routers
app.include_router(math.router)
app.include_router(multimodal.router)
app.include_router(websocket_router)
app.include_router(visualization_router)
app.include_router(workflow_router)
app.include_router(query_analysis_router)
app.include_router(chat_visualization.router)

# Initialize workflow engine and orchestration manager
workflow_engine = None
orchestration_manager = None
core_llm_agent = None

# Create a dictionary to store agent instances
agent_instances = {}

# Add default app configuration
app_config = {
    "initialize_agents": True,
    "initialize_viz_agent": True,
    "initialize_chat_analysis_agent": True
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global workflow_engine, orchestration_manager, core_llm_agent, agent_instances
    
    try:
        # Initialize workflow engine
        logger.info("Initializing workflow engine...")
        workflow_engine = WorkflowEngine()
        logger.info("WorkflowEngine initialized")
        
        # Initialize orchestration manager
        logger.info("Initializing orchestration manager...")
        orchestration_manager = get_orchestration_manager()
        await orchestration_manager.initialize()
        logger.info("Orchestration manager initialized")
        
        # Initialize Core LLM Agent
        logger.info("Initializing Core LLM Agent...")
        try:
            # Get environment settings for LMStudio server
            lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
            
            logger.info(f"LLM configuration: Using LMStudio server at {lmstudio_url}")
            
            # Test connection to LMStudio server
            from core.mistral.downloader import LMStudioValidator
            validator = LMStudioValidator(api_url=lmstudio_url)
            
            if not validator.check_connectivity():
                logger.warning(f"Could not connect to LMStudio server at {lmstudio_url}")
                logger.warning("Continuing without LLM agent - API will have limited functionality")
            else:
                logger.info(f"Successfully connected to LMStudio server")
                
                # Initialize the LLM agent with LMStudio config
                core_llm_agent = initialize_llm_agent()
                
                # Make the LLM agent accessible to other modules
                agent_instances["core_llm_agent"] = core_llm_agent
                logger.info(f"Initialized Core LLM Agent with LMStudio server at: {lmstudio_url}")
        except Exception as llm_error:
            logger.error(f"Error initializing Core LLM Agent: {str(llm_error)}")
            # Continue without the LLM agent for debugging purposes
            pass
        
        # Register core agents
        logger.info("Registering core agents...")
        register_core_agents()
        
        # Register agent instances with the registry
        registry = get_agent_registry()
        if core_llm_agent:
            registry.register_agent_instance("core_llm_agent", core_llm_agent)
            logger.info("Core LLM agent registered successfully")
        else:
            logger.error("Failed to initialize Core LLM Agent")
        
        logger.info("Core system agents registered")
        
        # Update agent status
        if core_llm_agent:
            registry.update_agent_status("core_llm_agent", "active")
        else:
            registry.update_agent_status("core_llm_agent", "error")
            
        registry.update_agent_status("math_computation_agent", "active")
        registry.update_agent_status("ocr_agent", "active")
        registry.update_agent_status("visualization_agent", "active")
        registry.update_agent_status("search_agent", "active")
        
        # Initialize agents as needed
        if app_config.get("initialize_agents", True):
            try:
                # Initialize other agents
                
                # Initialize visualization agent
                if app_config.get("initialize_viz_agent", True):
                    from visualization.agent.viz_agent import VisualizationAgent
                    app.state.viz_agent = VisualizationAgent({"storage_dir": "visualizations"})
                    agent_instances["visualization_agent"] = app.state.viz_agent
                    logger.info("Initialized Visualization Agent")
                
                # Initialize chat analysis agent
                if app_config.get("initialize_chat_analysis_agent", True):
                    from orchestration.agents.chat_analysis_agent import get_chat_analysis_agent
                    app.state.chat_analysis_agent = get_chat_analysis_agent()
                    agent_instances["chat_analysis_agent"] = app.state.chat_analysis_agent
                    logger.info("Initialized Chat Analysis Agent")
                    
            except Exception as e:
                logger.error(f"Error initializing agents: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on server shutdown."""
    global workflow_engine, orchestration_manager, core_llm_agent
    
    try:
        logger.info("Shutting down server components...")
        if workflow_engine and hasattr(workflow_engine, 'shutdown'):
            await workflow_engine.shutdown()
        
        if orchestration_manager and hasattr(orchestration_manager, 'shutdown'):
            await orchestration_manager.shutdown()
        
        if core_llm_agent:
            # Add any cleanup needed for Core LLM Agent
            pass
        
        logger.info("Server shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Mathematical Multimodal LLM System API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if the LLM agent is available
    llm_status = "available" if core_llm_agent else "unavailable"
    
    return {
        "status": "healthy",
        "components": {
            "llm_agent": llm_status,
            "workflow_engine": "available" if workflow_engine else "unavailable",
            "orchestration_manager": "available" if orchestration_manager else "unavailable"
        }
    }

@app.get("/agents/status")
async def agent_status():
    """Get the status of all agents."""
    registry = get_agent_registry()
    active_agents = registry.get_active_agents()
    
    return {
        "agents": active_agents,
        "total_count": len(active_agents),
        "capabilities": registry.get_all_capabilities()
    }

# Add endpoint to access the core_llm_agent
def get_core_llm_agent():
    """Get the core LLM agent instance."""
    global core_llm_agent, agent_instances
    if core_llm_agent:
        return core_llm_agent
    return agent_instances.get("core_llm_agent")

# Now that get_core_llm_agent is defined, import and include the ai_analysis_router
from api.rest.routes.ai_analysis import router as ai_analysis_router
app.include_router(ai_analysis_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def start_server(host="0.0.0.0", port=8000):
    """
    Start the FastAPI server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    import uvicorn
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

"""
FastAPI server for the Mathematical Multimodal LLM System.

This module sets up the FastAPI application with REST and WebSocket
endpoints for the system.
"""
import logging
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes import math, multimodal
from api.websocket.server import websocket_router
from api.rest.routes.visualization import router as visualization_router
from api.rest.routes.nlp_visualization import router as nlp_visualization_router
from .system_init import initialize_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Initialize system components
logger.info("Initializing Mathematical Multimodal LLM System")

# Must be initialized before any routes are loaded
# that depend on the services being registered
success = initialize_system()
if not success:
    logger.warning("System initialization had some issues. Some features may not work correctly.")

# Create FastAPI application
app = FastAPI(
    title="Mathematical Multimodal LLM System",
    description="API for processing mathematical content across multiple modalities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with actual origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(math.router)
app.include_router(multimodal.router)
app.include_router(websocket_router)
app.include_router(visualization_router)
app.include_router(nlp_visualization_router)

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Mathematical Multimodal LLM System API",
        "version": "1.0.0",
        "status": "running",
        "llm_config": {
            "use_lmstudio": os.environ.get('USE_LMSTUDIO', '1') == '1',
            "lmstudio_url": os.environ.get('LMSTUDIO_URL', 'http://127.0.0.1:1234'),
            "lmstudio_model": os.environ.get('LMSTUDIO_MODEL', 'mistral-7b-instruct-v0.3')
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    # Log configuration information
    logger.info(f"Starting server on {host}:{port}")
    
    # Log LMStudio configuration
    logger.info(f"LMStudio configuration:")
    logger.info(f"- URL: {os.environ.get('LMSTUDIO_URL', 'http://127.0.0.1:1234')}")
    logger.info(f"- Model: {os.environ.get('LMSTUDIO_MODEL', 'mistral-7b-instruct-v0.3')}")
    logger.info(f"- Enabled: {os.environ.get('USE_LMSTUDIO', '1') == '1'}")
    
    # Start the server
    try:
        uvicorn.run(
            app=app,
            host=host,
            port=port,
            log_level="debug" if os.environ.get('DEBUG', '0') == '1' else "info"
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Startup script for the Mathematical Multimodal LLM System server.

This script initializes all components and starts the API server with a
single shared event loop to prevent "attached to a different loop" errors.
"""

import os
import sys
import logging
import argparse
import asyncio
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

# Ensure all required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
os.makedirs("models/cache", exist_ok=True)

class SingletonEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """A policy that ensures a single event loop is used throughout the application."""
    
    def __init__(self):
        super().__init__()
        self._shared_loop = None
    
    def get_event_loop(self):
        """Return the shared event loop, creating one if necessary."""
        if self._shared_loop is None:
            self._shared_loop = self.new_event_loop()
        return self._shared_loop
    
    def set_event_loop(self, loop):
        """Set the event loop as the shared loop."""
        self._shared_loop = loop
        super().set_event_loop(loop)

class CustomUvicornServer(uvicorn.Server):
    """Custom Uvicorn server that allows for graceful shutdown."""
    
    def install_signal_handlers(self):
        """Override to avoid installing signal handlers that conflict with the main loop."""
        pass

async def run_app(host, port):
    """Run the FastAPI app with a custom server."""
    from api.rest.server import app
    
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = CustomUvicornServer(config)
    
    await server.serve()

def main():
    """Main function to initialize and start the server."""
    parser = argparse.ArgumentParser(description='Start the Mathematical Multimodal LLM System server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['API_HOST'] = args.host
    os.environ['API_PORT'] = str(args.port)
    os.environ['DEBUG'] = '1' if args.debug else '0'
    os.environ['VISUALIZATION_DIR'] = 'visualizations'
    
    # Install the singleton event loop policy
    asyncio.set_event_loop_policy(SingletonEventLoopPolicy())
    
    # Create and set the shared event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Store the loop in a global variable for other modules to access
    setattr(asyncio, '_mathllm_shared_loop', loop)
    
    logger.info(f"Starting Mathematical Multimodal LLM System on {args.host}:{args.port}")
    
    try:
        # Run the application with the shared event loop
        loop.run_until_complete(run_app(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
    finally:
        # Clean up tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Run the event loop until all tasks are cancelled
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        # Close the event loop
        loop.close()

if __name__ == "__main__":
    main()

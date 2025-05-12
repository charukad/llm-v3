#!/usr/bin/env python3
"""
Startup script for the Mathematical Multimodal LLM System server.

This script initializes all components and starts the API server.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the project root directory to PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

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
    
    logger.info(f"Starting Mathematical Multimodal LLM System on {args.host}:{args.port}")
    
    # Import and start the server
    from api.rest.server import app
    import uvicorn
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 
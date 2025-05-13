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
    
    # LMStudio configuration
    parser.add_argument('--lmstudio-url', type=str, default='http://127.0.0.1:1234', 
                      help='URL of the LMStudio server')
    parser.add_argument('--lmstudio-model', type=str, default='mistral-7b-instruct-v0.3', 
                      help='Model name in LMStudio')
    parser.add_argument('--no-lmstudio', action='store_true', 
                      help='Disable LMStudio integration (use local models)')
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['API_HOST'] = args.host
    os.environ['API_PORT'] = str(args.port)
    os.environ['DEBUG'] = '1' if args.debug else '0'
    os.environ['VISUALIZATION_DIR'] = 'visualizations'
    
    # LMStudio environment variables
    os.environ['LMSTUDIO_URL'] = args.lmstudio_url
    os.environ['LMSTUDIO_MODEL'] = args.lmstudio_model
    os.environ['USE_LMSTUDIO'] = '0' if args.no_lmstudio else '1'
    
    logger.info(f"Starting Mathematical Multimodal LLM System on {args.host}:{args.port}")
    if not args.no_lmstudio:
        logger.info(f"Using LMStudio at {args.lmstudio_url} with model {args.lmstudio_model}")
    
    # Import server module
    from api.rest.server import start_server
    
    # Start the server
    start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()

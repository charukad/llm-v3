#!/usr/bin/env python3
"""
Launch script for the Mathematical Multimodal LLM System with LMStudio integration.

This script configures and starts the system with LMStudio as the inference backend.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/lmstudio_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

def check_lmstudio_server(url: str, retries: int = 3, delay: int = 2) -> bool:
    """
    Check if LMStudio server is running and accessible.
    
    Args:
        url: LMStudio server URL
        retries: Number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        True if LMStudio server is running, False otherwise
    """
    url = url.rstrip('/')
    models_url = f"{url}/v1/models"
    
    for attempt in range(retries):
        try:
            logger.info(f"Checking LMStudio server at {url} (attempt {attempt+1}/{retries})")
            response = requests.get(models_url, timeout=5)
            
            if response.status_code == 200:
                models = response.json()
                logger.info(f"LMStudio server is running. Available models: {models}")
                return True
            else:
                logger.warning(f"LMStudio server responded with status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Error connecting to LMStudio server: {e}")
        
        if attempt < retries - 1:
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    return False

def launch_system(args: argparse.Namespace) -> None:
    """
    Launch the Mathematical Multimodal LLM System.
    
    Args:
        args: Command-line arguments
    """
    # Check if LMStudio server is running
    if not check_lmstudio_server(args.lmstudio_url):
        logger.error(f"LMStudio server is not accessible at {args.lmstudio_url}")
        logger.error("Please start LMStudio server before running this script")
        logger.error("Make sure the model is loaded and the API is enabled in LMStudio")
        sys.exit(1)
    
    # Define command to run the server
    cmd = [
        sys.executable,
        "run_server.py",
        "--host", args.host,
        "--port", str(args.port),
        "--lmstudio-url", args.lmstudio_url,
        "--lmstudio-model", args.lmstudio_model
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    # Set environment variables for configuration
    env = os.environ.copy()
    env["LMSTUDIO_URL"] = args.lmstudio_url
    env["LMSTUDIO_MODEL"] = args.lmstudio_model
    env["USE_LMSTUDIO"] = "1"
    
    # Launch the server
    logger.info(f"Launching Mathematical Multimodal LLM System with LMStudio integration")
    logger.info(f"LMStudio URL: {args.lmstudio_url}")
    logger.info(f"LMStudio Model: {args.lmstudio_model}")
    logger.info(f"Server will be available at http://{args.host}:{args.port}")
    
    try:
        # Run the command
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running the server: {e}")

def main() -> None:
    """Parse arguments and launch the system."""
    parser = argparse.ArgumentParser(
        description="Launch Mathematical Multimodal LLM System with LMStudio integration"
    )
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0",
                      help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port to bind the server to")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode")
    
    # LMStudio configuration
    parser.add_argument("--lmstudio-url", default="http://127.0.0.1:1234",
                      help="URL of the LMStudio server")
    parser.add_argument("--lmstudio-model", default="mistral-7b-instruct-v0.3",
                      help="Model name in LMStudio")
    
    args = parser.parse_args()
    
    # Launch the system
    launch_system(args)

if __name__ == "__main__":
    main() 
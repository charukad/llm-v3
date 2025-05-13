#!/usr/bin/env python3
"""
Server Health Check Script for the Mathematical Multimodal LLM System.

This script checks if all the necessary components are properly initialized
and ready to handle requests.
"""
import os
import sys
import logging
import requests
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def check_server_health(url: str, retries: int = 3, delay: int = 2) -> bool:
    """
    Check if the server is healthy.
    
    Args:
        url: Server URL
        retries: Number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        True if the server is healthy, False otherwise
    """
    health_url = f"{url}/health"
    
    for attempt in range(retries):
        try:
            logger.info(f"Checking server health at {health_url} (attempt {attempt+1}/{retries})")
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Server health check successful: {health_data}")
                return True
            else:
                logger.warning(f"Server health check failed with status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Error connecting to server: {e}")
        
        if attempt < retries - 1:
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    return False

def check_lmstudio_integration(url: str) -> bool:
    """
    Check if LMStudio integration is configured.
    
    Args:
        url: Server URL
        
    Returns:
        True if LMStudio integration is configured, False otherwise
    """
    try:
        logger.info(f"Checking LMStudio integration at {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            llm_config = data.get("llm_config", {})
            
            if llm_config.get("use_lmstudio"):
                logger.info(f"LMStudio integration is enabled")
                logger.info(f"LMStudio URL: {llm_config.get('lmstudio_url')}")
                logger.info(f"LMStudio model: {llm_config.get('lmstudio_model')}")
                return True
            else:
                logger.warning("LMStudio integration is disabled")
        else:
            logger.warning(f"Failed to check LMStudio integration: status code {response.status_code}")
    except Exception as e:
        logger.error(f"Error checking LMStudio integration: {e}")
    
    return False

def send_test_query(url: str) -> bool:
    """
    Send a test query to the server.
    
    Args:
        url: Server URL
        
    Returns:
        True if the test query was successful, False otherwise
    """
    test_url = f"{url}/math/solve"
    test_data = {"query": "What is the derivative of x^2?"}
    
    try:
        logger.info(f"Sending test query to {test_url}")
        response = requests.post(test_url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Test query successful: {result}")
            return True
        else:
            logger.warning(f"Test query failed with status code {response.status_code}")
            if response.text:
                logger.warning(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Error sending test query: {e}")
    
    return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check server health")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--test-query", action="store_true", help="Send a test query")
    args = parser.parse_args()
    
    server_url = f"http://{args.host}:{args.port}"
    
    logger.info(f"Checking server at {server_url}")
    
    # Check server health
    if check_server_health(server_url):
        logger.info("✅ Server is healthy")
    else:
        logger.error("❌ Server health check failed")
        return 1
    
    # Check LMStudio integration
    if check_lmstudio_integration(server_url):
        logger.info("✅ LMStudio integration is configured")
    else:
        logger.warning("⚠️ LMStudio integration check failed")
    
    # Send test query if requested
    if args.test_query:
        if send_test_query(server_url):
            logger.info("✅ Test query successful")
        else:
            logger.warning("⚠️ Test query failed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
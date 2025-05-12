#!/usr/bin/env python3
"""
Debug script for testing the AI analysis agent connection with the LLM.
This script sends a test query to the AI analysis endpoint and logs detailed information
about the connection between the agent and the LLM.
"""

import requests
import json
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"debug_ai_analysis.log")
    ]
)

logger = logging.getLogger("debug_ai_analysis")

def test_ai_analysis(query="What is 2+2?"):
    """
    Test the AI analysis endpoint with a simple query.
    
    Args:
        query: The query to analyze (default: "What is 2+2?")
    """
    logger.info(f"Testing AI analysis with query: {query}")
    
    # Endpoint URL
    api_url = "http://localhost:8000/ai-analysis/analyze"
    
    # Request payload
    payload = {
        "query": query,
        "context_id": None,
        "conversation_id": None
    }
    
    # First check if server is running
    try:
        health_check = requests.get("http://localhost:8000/health", timeout=10)  # Increased timeout for health check
        if health_check.status_code == 200:
            health_data = health_check.json()
            logger.info(f"Server health check: {health_data}")
            
            # Check LLM agent status
            llm_status = health_data.get("components", {}).get("llm_agent", "unknown")
            logger.info(f"LLM agent status: {llm_status}")
            
            if llm_status != "available":
                logger.warning("LLM agent is not available - analysis may use fallback mechanisms")
        else:
            logger.error(f"Health check failed with status: {health_check.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to server: {e}")
        return False
    
    # Make the request
    try:
        logger.info(f"Sending request to {api_url}")
        start_time = time.time()
        
        response = requests.post(
            api_url,
            json=payload,
            timeout=300  # Increased to 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Response received in {duration:.2f} seconds")
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                logger.info(f"Response: {json.dumps(response_data, indent=2)}")
                
                # Check if analysis used fallback mechanisms
                analysis = response_data.get("analysis", {})
                ai_source = analysis.get("ai_source", "unknown")
                fallback = analysis.get("fallback", False)
                
                if fallback or ai_source != "real_core_llm":
                    logger.warning(f"Analysis used fallback mechanism: ai_source={ai_source}, fallback={fallback}")
                    if "error" in analysis:
                        logger.error(f"Error in analysis: {analysis['error']}")
                else:
                    logger.info("Analysis was performed by the real LLM agent")
                
                return True
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response: {response.text}")
                return False
        else:
            logger.error(f"Request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False

if __name__ == "__main__":
    # Get query from command line argument if provided
    query = sys.argv[1] if len(sys.argv) > 1 else "Solve x^2 + 5x + 6 = 0 step by step"
    
    logger.info("Starting AI analysis test")
    success = test_ai_analysis(query)
    
    if success:
        logger.info("Test completed successfully")
    else:
        logger.error("Test failed")
        sys.exit(1) 
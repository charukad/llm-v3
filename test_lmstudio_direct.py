#!/usr/bin/env python3
"""
Direct LMStudio test script to verify LMStudio connection.
"""
import os
import logging
import sys
import requests
import json
from core.agent.llm_agent import CoreLLMAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def test_direct_lmstudio():
    """Test direct connection to LMStudio API."""
    lmstudio_url = os.environ.get('LMSTUDIO_URL', 'http://127.0.0.1:1234')
    
    logger.info(f"Testing direct connection to LMStudio at {lmstudio_url}")
    
    try:
        response = requests.get(f"{lmstudio_url}/v1/models")
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Connected to LMStudio API. Available models: {models}")
        else:
            logger.error(f"LMStudio API responded with status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Could not connect to LMStudio API: {str(e)}")
        return False
    
    logger.info("Testing a simple generation request")
    
    try:
        payload = {
            "model": "mistral-7b-instruct-v0.3",
            "prompt": "Solve the equation x^2 + 5x + 6 = 0",
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        response = requests.post(f"{lmstudio_url}/v1/completions", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Generation successful")
            logger.info(f"Generated text: {result.get('choices', [{}])[0].get('text', 'No text')}")
            return True
        else:
            logger.error(f"Generation failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error in generation request: {str(e)}")
        return False

def test_core_llm_agent():
    """Test CoreLLMAgent with LMStudio."""
    logger.info("Testing CoreLLMAgent with LMStudio")
    
    try:
        agent = CoreLLMAgent()
        result = agent.generate_response("Solve x^2 + 5x + 6 = 0")
        
        if result.get("success", False):
            logger.info("CoreLLMAgent generation successful")
            logger.info(f"Generated response: {result.get('response', 'No response')}")
            return True
        else:
            logger.error(f"CoreLLMAgent generation failed: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error in CoreLLMAgent test: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting LMStudio connection tests")
    
    direct_result = test_direct_lmstudio()
    agent_result = test_core_llm_agent()
    
    if direct_result and agent_result:
        logger.info("All tests passed - LMStudio connection is working correctly")
        sys.exit(0)
    else:
        logger.error("Some tests failed - check logs for details")
        sys.exit(1) 
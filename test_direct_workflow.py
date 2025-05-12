#!/usr/bin/env python3
"""
Direct test of the workflow/process/text endpoint.
"""
import time
import json
import sys
import logging
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_workflow_direct(query):
    """Test the workflow/process/text endpoint directly."""
    
    # Endpoint URL
    endpoint_url = "http://localhost:8000/workflow/process/text"
    
    logger.info(f"Sending query to workflow endpoint: {query}")
    start_time = time.time()
    
    # Prepare payload
    payload = {
        "text": query,
        "context_id": None,
        "conversation_id": None,
        "workflow_options": {
            "generate_visualization": False
        }
    }
    
    try:
        # Send request to workflow endpoint
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=300  # 300 second timeout (5 minutes)
        )
        
        # Check if request was successful
        if response.status_code != 200:
            logger.error(f"Error from workflow endpoint: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {
                "success": False,
                "error": f"Workflow endpoint returned status code: {response.status_code}",
                "response_text": response.text
            }
        
        # Parse response
        response_data = response.json()
        
        # Calculate time taken
        process_time = time.time() - start_time
        logger.info(f"Processing completed in {process_time:.2f} seconds")
        
        return {
            "success": True,
            "response": response_data,
            "process_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
    
    logger.info(f"Running test with query: {query}")
    
    # First, check if server is up
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        logger.info(f"Server health check: {health_response.status_code}")
        if health_response.status_code != 200:
            print(f"\nERROR: Server health check failed: {health_response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Server is not running or not accessible: {str(e)}")
        sys.exit(1)
    
    result = test_workflow_direct(query)
    
    if result.get("success", False):
        print("\nSUCCESS: Workflow test completed")
        print(f"Process time: {result.get('process_time'):.2f}s")
        
        # Print response details
        response_data = result.get("response", {})
        output = response_data.get("output", "No output")
        context_id = response_data.get("context_id", "No context ID")
        
        print(f"\nOutput: {output}")
        print(f"Context ID: {context_id}")
        
        # Print full response for debugging
        print("\nFull response:")
        print(json.dumps(response_data, indent=2))
        
        # Check for any additional data
        additional_keys = [k for k in response_data.keys() if k not in ["output", "context_id"]]
        if additional_keys:
            print("\nAdditional data:")
            for key in additional_keys:
                print(f"- {key}: {response_data[key]}")
    else:
        print("\nFAILURE: Could not complete workflow test")
        print(f"Error: {result.get('error', 'Unknown error')}")
        if "response_text" in result:
            print(f"Response: {result.get('response_text')}") 
#!/usr/bin/env python3
"""
Test script for the chat visualization functionality.

This script tests the ability to extract plot information from natural language
requests and generate the appropriate visualizations.
"""
import requests
import json
import sys
import logging
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_chat_visualization(text: str, base_url: str = "http://localhost:8000"):
    """
    Test the chat visualization functionality with the given text.
    
    Args:
        text: The text to analyze for visualization requests
        base_url: Base URL of the API
    
    Returns:
        Dictionary with the response data
    """
    logger.info(f"Testing chat visualization with text: {text}")
    
    # Endpoint URL for chat visualization
    url = f"{base_url}/chat/visualize"
    
    # Prepare payload
    payload = {
        "text": text,
        "generate_immediately": True
    }
    
    try:
        # Send request
        response = requests.post(url, json=payload, timeout=30)
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Error response: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"API returned error code {response.status_code}",
                "details": response.text
            }
        
        # Parse response
        result = response.json()
        
        # Print results
        if result.get("is_visualization_request", False):
            logger.info(f"Successfully identified visualization request!")
            logger.info(f"Plot type: {result.get('plot_type')}")
            logger.info(f"Parameters: {json.dumps(result.get('parameters', {}), indent=2)}")
            
            # If visualization was generated
            if "visualization" in result and result["visualization"].get("success", False):
                logger.info("Visualization generated successfully")
                logger.info(f"URL: {result['visualization'].get('url')}")
            else:
                logger.warning("Visualization request was identified but generation failed")
                if "visualization" in result:
                    logger.warning(f"Error: {result['visualization'].get('error')}")
        else:
            logger.info(f"Text was not identified as a visualization request")
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def test_workflow_integration(text: str, base_url: str = "http://localhost:8000"):
    """
    Test the integration with the workflow system.
    
    Args:
        text: The text to process
        base_url: Base URL of the API
    
    Returns:
        Dictionary with the response data
    """
    logger.info(f"Testing workflow integration with text: {text}")
    
    # Endpoint URL for workflow processing
    url = f"{base_url}/workflow/process/text"
    
    # Prepare payload
    payload = {
        "text": text
    }
    
    try:
        # Send request
        response = requests.post(url, json=payload, timeout=30)
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Error response: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"API returned error code {response.status_code}",
                "details": response.text
            }
        
        # Parse response
        result = response.json()
        
        # Check for visualization information
        if "result" in result and "visualizations" in result["result"]:
            logger.info("Workflow generated visualizations!")
            for i, viz in enumerate(result["result"]["visualizations"]):
                logger.info(f"Visualization {i+1}: {viz.get('url')}")
                
        return result
        
    except Exception as e:
        logger.error(f"Error processing workflow request: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test chat visualization functionality")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", 
                        help="Base URL of the API")
    parser.add_argument("--workflow", action="store_true", 
                        help="Test workflow integration instead of direct API")
    
    args = parser.parse_args()
    
    # If no text provided, use examples
    if not args.text:
        examples = [
            "Can you plot sin(x) for me?",
            "I want to see a 3D visualization of z = sin(x) * cos(y)",
            "Please plot the function f(x) = x^2 - 2*x + 3 from -5 to 5",
            "Show me a graph of sin(x) and cos(x) on the same plot",
            "Create a 3D helix with x = cos(t), y = sin(t), z = t"
        ]
        
        for example in examples:
            print("\n" + "="*80)
            print(f"TESTING: {example}")
            print("="*80)
            
            # Run test
            if args.workflow:
                result = test_workflow_integration(example, args.base_url)
            else:
                result = test_chat_visualization(example, args.base_url)
            
            # Print full result for debugging
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            print("\n")
            
            # Wait a bit between tests
            time.sleep(1)
    else:
        # Test with provided text
        if args.workflow:
            result = test_workflow_integration(args.text, args.base_url)
        else:
            result = test_chat_visualization(args.text, args.base_url)
        
        # Print full result for debugging
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 
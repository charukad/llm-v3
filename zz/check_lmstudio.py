#!/usr/bin/env python3
"""
Utility script to check if LMStudio is available with the required model.

This script verifies the connection to LMStudio and checks if the specified
model is available. It also performs a simple test query to ensure the model
can generate responses.
"""

import argparse
import logging
import os
import sys
import time
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def check_lmstudio_api(url: str) -> dict:
    """
    Check if LMStudio API is available.
    
    Args:
        url: LMStudio API URL
        
    Returns:
        Status information
    """
    url = url.rstrip('/')
    models_url = f"{url}/v1/models"
    
    try:
        logger.info(f"Checking LMStudio API at {url}")
        response = requests.get(models_url, timeout=5)
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "LMStudio API is available",
                "models": response.json()
            }
        else:
            return {
                "status": "error",
                "message": f"LMStudio API responded with status code {response.status_code}",
                "raw_response": response.text
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error connecting to LMStudio API: {str(e)}"
        }

def check_model_availability(url: str, model_name: str) -> dict:
    """
    Check if the specified model is available in LMStudio.
    
    Args:
        url: LMStudio API URL
        model_name: Name of the model to check
        
    Returns:
        Status information
    """
    api_status = check_lmstudio_api(url)
    
    if api_status["status"] == "error":
        return api_status
    
    # Check if the model is in the list
    try:
        models = api_status["models"]
        available_models = [model["id"] for model in models.get("data", [])]
        
        if model_name in available_models:
            return {
                "status": "success",
                "message": f"Model '{model_name}' is available",
                "available_models": available_models
            }
        else:
            return {
                "status": "error",
                "message": f"Model '{model_name}' is not available in LMStudio",
                "available_models": available_models
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking model availability: {str(e)}"
        }

def test_model_inference(url: str, model_name: str) -> dict:
    """
    Test the model with a simple math query.
    
    Args:
        url: LMStudio API URL
        model_name: Name of the model to test
        
    Returns:
        Test result information
    """
    url = url.rstrip('/')
    completions_url = f"{url}/v1/completions"
    
    # Simple math query
    test_prompt = "What is the square root of 144? Explain step by step."
    
    try:
        logger.info(f"Testing model '{model_name}' with a simple math query")
        
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "max_tokens": 100,
            "temperature": 0.1,
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post(completions_url, json=payload, timeout=30)
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["text"] if "choices" in result and result["choices"] else ""
            
            return {
                "status": "success",
                "message": "Model inference test successful",
                "time_taken": f"{generation_time:.2f} seconds",
                "generated_text": generated_text
            }
        else:
            return {
                "status": "error",
                "message": f"Model inference failed with status code {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during model inference: {str(e)}"
        }

def main():
    """Run the LMStudio check."""
    parser = argparse.ArgumentParser(description="Check LMStudio availability and run a test query")
    parser.add_argument("--url", default="http://127.0.0.1:1234", help="LMStudio API URL")
    parser.add_argument("--model", default="mistral-7b-instruct-v0.3", help="Model name in LMStudio")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()
    
    # Step 1: Check LMStudio API
    api_status = check_lmstudio_api(args.url)
    
    if args.json:
        results = {"api_check": api_status}
    else:
        print("\n=== LMStudio API Check ===")
        print(f"Status: {api_status['status']}")
        print(f"Message: {api_status['message']}")
        
        if api_status["status"] == "success" and "models" in api_status:
            models = api_status["models"]
            print(f"Available models: {[model['id'] for model in models.get('data', [])]}")
    
    # If API is available, check model availability
    if api_status["status"] == "success":
        model_status = check_model_availability(args.url, args.model)
        
        if args.json:
            results["model_check"] = model_status
        else:
            print("\n=== Model Availability Check ===")
            print(f"Status: {model_status['status']}")
            print(f"Message: {model_status['message']}")
            
            if "available_models" in model_status:
                print(f"All available models: {model_status['available_models']}")
        
        # If model is available, test inference
        if model_status["status"] == "success":
            inference_status = test_model_inference(args.url, args.model)
            
            if args.json:
                results["inference_test"] = inference_status
            else:
                print("\n=== Model Inference Test ===")
                print(f"Status: {inference_status['status']}")
                print(f"Message: {inference_status['message']}")
                
                if inference_status["status"] == "success":
                    print(f"Time taken: {inference_status['time_taken']}")
                    print(f"Generated text: {inference_status['generated_text']}")
    
    # Output JSON if requested
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Return an exit code
    if api_status["status"] == "error":
        sys.exit(1)
    elif "model_check" in locals() and model_status["status"] == "error":
        sys.exit(2)
    elif "inference_test" in locals() and inference_status["status"] == "error":
        sys.exit(3)
    else:
        print("\n✅ LMStudio is ready to use with the Mathematical LLM System")
        sys.exit(0)

if __name__ == "__main__":
    main() 
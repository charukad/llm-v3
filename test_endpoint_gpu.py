#!/usr/bin/env python3
"""
Test script for the /workflow/process/text endpoint with GPU acceleration.
"""

import os
import requests
import json
import time
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the workflow/process/text endpoint with GPU acceleration")
    parser.add_argument("--query", type=str, required=True,
                      help="The query to send to the endpoint")
    parser.add_argument("--model-layers", type=int, default=128,
                      help="Number of GPU layers to use")
    parser.add_argument("--use-mps", action="store_true",
                      help="Use Apple Metal Performance Shaders (MPS) for acceleration")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port where the API server is running")
    
    return parser.parse_args()

def main():
    """Main function to run the test."""
    args = parse_args()
    
    # Set environment variables for GPU acceleration
    os.environ["MODEL_LAYERS"] = str(args.model_layers)
    if args.use_mps:
        os.environ["USE_MPS"] = "1"
        print(f"Using Apple Metal (MPS) acceleration with {args.model_layers} layers")
    else:
        print(f"Using {args.model_layers} GPU layers")
    
    # Prepare the API request
    url = f"http://localhost:{args.port}/workflow/process/text"
    data = {
        "text": args.query,
        "conversation_id": f"test-gpu-{int(time.time())}"
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\n" + "="*70)
    print(f"QUERY: {args.query}")
    print("="*70)
    
    # Send the request and measure time
    start_time = time.time()
    print(f"Sending request to {url}...")
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=300)
        elapsed_time = time.time() - start_time
        
        print(f"Request completed in {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("\nRESPONSE:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\nError: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {str(e)}")
    
    print("\n" + "="*70)
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script for the Natural Language Visualization API endpoint.
"""

import requests
import json
import os
import argparse
from pprint import pprint

def test_nlp_visualization(prompt, server_url="http://localhost:8000"):
    """
    Test the NLP visualization endpoint with a given prompt.
    
    Args:
        prompt: Natural language description of the visualization
        server_url: URL of the API server
    """
    # Endpoint for NLP visualization
    endpoint = f"{server_url}/nlp-visualization"
    
    # Request data
    request_data = {
        "prompt": prompt
    }
    
    print(f"Sending NLP visualization request to {endpoint}")
    print(f"Prompt: {prompt}")
    print("\nSending request...")
    
    # Make the request
    try:
        response = requests.post(endpoint, json=request_data)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\nResponse:")
            print(f"Success: {result.get('success')}")
            print(f"Visualization Type: {result.get('visualization_type')}")
            
            # Check if we have a file path
            if result.get('file_path'):
                print(f"File Path: {result.get('file_path')}")
                
                # Check if the file exists
                if os.path.exists(result.get('file_path')):
                    print(f"Visualization saved to: {result.get('file_path')}")
                else:
                    print(f"Warning: File not found at {result.get('file_path')}")
            
            # Check if we have base64 image data
            if result.get('base64_image'):
                print(f"Base64 Image: {len(result.get('base64_image'))} bytes")
            
            # Check for errors
            if not result.get('success') and result.get('error'):
                print(f"\nError: {result.get('error')}")
            
            # Show LLM analysis
            if result.get('llm_analysis'):
                print("\nLLM Analysis:")
                pprint(result.get('llm_analysis'))
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error making request: {e}")
        return None

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test the NLP visualization endpoint")
    parser.add_argument("--prompt", type=str, help="Visualization prompt", 
                        default="Create a scatter plot with these points: (1,3), (2,5), (3,4), (4,7), (5,8), (6,10)")
    parser.add_argument("--url", type=str, help="Server URL", default="http://localhost:8000")
    args = parser.parse_args()
    
    test_nlp_visualization(args.prompt, args.url)

if __name__ == "__main__":
    main() 
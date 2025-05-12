#!/usr/bin/env python3
"""
Simple script to test the workflow/process/text API endpoint.
"""

import requests
import json
import sys
import time

def call_workflow_api(query_text):
    """Call the workflow API with a text query."""
    url = "http://localhost:8000/workflow/process/text"
    headers = {"Content-Type": "application/json"}
    
    # Important: The text parameter needs to be embedded in a JSON object with a "text" key
    # This is because the FastAPI endpoint uses Body(..., embed=True)
    payload = {"text": query_text}
    
    print(f"Sending request to {url}")
    print(f"Request payload: {json.dumps(payload)}")
    
    try:
        # Set a timeout to avoid hanging
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        print(f"Status code: {response.status_code}")
        return response.json()
    except requests.exceptions.Timeout:
        print("Request timed out after 10 seconds. The server might be busy or not running.")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error. Make sure the server is running at http://localhost:8000")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

if __name__ == "__main__":
    query = "Show me sin(x)*cos(x) and its integral"
    
    # Use command line argument if provided
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    # First check if the server is reachable
    try:
        health_check = requests.get("http://localhost:8000/health", timeout=2)
        if health_check.status_code == 200:
            print("Server is up and running!")
        else:
            print(f"Server returned unexpected status: {health_check.status_code}")
    except requests.exceptions.RequestException:
        print("Could not reach the server. Make sure it's running on http://localhost:8000")
        sys.exit(1)
    
    result = call_workflow_api(query)
    
    if result:
        print("\nAPI Response:")
        print(json.dumps(result, indent=2)) 
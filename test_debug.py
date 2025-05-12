#!/usr/bin/env python3
"""
Debug script for the LLM agent issue.
"""

import requests
import json
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_health(base_url="http://localhost:8000"):
    """Test server health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        print("✓ Server health check passed")
        return True
    except Exception as e:
        print(f"✗ Server health check failed: {str(e)}")
        return False

def check_api_endpoints(base_url="http://localhost:8000"):
    """Check available API endpoints"""
    endpoints = [
        "/",
        "/health",
        "/workflow",
        "/workflow/process",
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"Endpoint {endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {json.dumps(response.json(), indent=2)}")
        except requests.exceptions.RequestException as e:
            print(f"Endpoint {endpoint}: Error - {str(e)}")

def test_simple_query(base_url="http://localhost:8000"):
    """Test a simple query to debug the LLM agent"""
    test_data = {
        "input_type": "text",
        "content": "What is 2+2?",
        "content_type": "text/plain",
        "context_id": None,
        "conversation_id": None
    }
    
    try:
        print("\nTesting simple query...")
        response = requests.post(
            f"{base_url}/workflow/process",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Initial Response: {json.dumps(result, indent=2)}")
        
        if "workflow_id" in result:
            print(f"Checking workflow status...")
            status_response = requests.get(f"{base_url}/workflow/{result['workflow_id']}/status")
            status_data = status_response.json()
            print(f"Status: {json.dumps(status_data, indent=2)}")
            
            if status_data.get("status") == "completed":
                result_response = requests.get(f"{base_url}/workflow/{result['workflow_id']}/result")
                result_data = result_response.json()
                print(f"Result: {json.dumps(result_data, indent=2)}")
                
                if "error" in result_data:
                    print(f"⚠️ Error detected: {result_data['error']}")
                    return False
                return True
        return False
        
    except Exception as e:
        print(f"Error testing simple query: {str(e)}")
        return False

if __name__ == "__main__":
    base_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Testing server at {base_url}...")
    
    if not test_health(base_url):
        sys.exit(1)
        
    print("\nChecking available API endpoints...")
    check_api_endpoints(base_url)
    
    print("\nTesting simple query...")
    test_simple_query(base_url) 
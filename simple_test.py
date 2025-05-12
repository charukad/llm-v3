#!/usr/bin/env python3
"""
Simple script to test the AI Analysis endpoint.
"""
import requests
import json
import sys

def test_endpoint(query):
    """Test the AI Analysis endpoint with a query."""
    url = "http://localhost:8000/ai-analysis/analyze"
    
    print(f"Sending query: {query}")
    
    try:
        response = requests.post(
            url,
            json={"query": query},
            timeout=60  # 60 second timeout
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            
            # Check if response came from real CoreLLM
            if result.get("analysis", {}).get("ai_source") == "real_core_llm":
                print("✅ SUCCESS: Response came from real CoreLLM")
            else:
                print("⚠️ WARNING: Response came from fallback mechanism")
                print(f"Error: {result.get('analysis', {}).get('error', 'Unknown error')}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 2+235?"
    
    test_endpoint(query) 
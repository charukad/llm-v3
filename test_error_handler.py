#!/usr/bin/env python3
"""
Test script to verify the error handler by triggering different types of errors.
Run this script while the server is running to see error handling in action.
"""

import requests
import json
import sys

# Configuration
SERVER_URL = "http://localhost:8000"
MATH_ENDPOINT = "/math/query"

def test_error_handler():
    """Run a series of tests to trigger different error responses"""
    print("Running Error Handler Tests")
    print("=" * 50)
    
    tests = [
        {
            "name": "1. Missing required field",
            "endpoint": f"{SERVER_URL}{MATH_ENDPOINT}",
            "payload": {},  # Missing required 'query' field
            "expected_status": 422,  # Validation error
        },
        {
            "name": "2. Malformed JSON",
            "endpoint": f"{SERVER_URL}{MATH_ENDPOINT}",
            "raw_data": "{bad json",
            "expected_status": 422,  # Updated from 400 to 422 - FastAPI treats this as validation error
        },
        {
            "name": "3. Invalid URL path",
            "endpoint": f"{SERVER_URL}/nonexistent/endpoint",
            "payload": {"query": "test"},
            "expected_status": 404,  # Not found
        },
        {
            "name": "4. Division by zero (potential backend error)",
            "endpoint": f"{SERVER_URL}{MATH_ENDPOINT}",
            "payload": {"query": "Calculate 1/0"},
            "expected_status": 200,  # Should still return 200 with a workflow ID
        },
        {
            "name": "5. Complex query (stress test)",
            "endpoint": f"{SERVER_URL}{MATH_ENDPOINT}",
            "payload": {"query": "Find the eigenvalues of a 10×10 Hilbert matrix"},
            "expected_status": 200,  # Should still return 200 with a workflow ID
        }
    ]
    
    results = []
    
    # Run each test
    for test in tests:
        print(f"\nRunning test: {test['name']}")
        print("-" * 50)
        
        try:
            headers = {"Content-Type": "application/json", "X-Request-ID": f"test-{len(results)+1}"}
            
            if "raw_data" in test:
                # Send intentionally malformed data
                response = requests.post(
                    test["endpoint"],
                    headers=headers,
                    data=test["raw_data"]
                )
            else:
                # Send normal JSON payload
                response = requests.post(
                    test["endpoint"],
                    headers=headers,
                    json=test["payload"]
                )
            
            # Print basic response info
            print(f"Status: {response.status_code}")
            
            # Try to print response body
            try:
                body = response.json()
                print(f"Response: {json.dumps(body, indent=2)}")
                
                # Check for error structure
                if "error" in body:
                    print(f"Error Type: {body['error'].get('type')}")
                    print(f"Error Message: {body['error'].get('message')}")
                    print(f"Request ID: {body['error'].get('request_id')}")
            except:
                print(f"Raw response: {response.text[:200]}...")
            
            # Determine test result
            success = response.status_code == test["expected_status"]
            results.append(success)
            
            print(f"Test result: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            print(f"Exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for i, test in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test['name']}: {status}")
    
    passed = results.count(True)
    total = len(results)
    print(f"\nPassed {passed} of {total} tests ({passed/total*100:.0f}%)")
    
    return all(results)

if __name__ == "__main__":
    success = test_error_handler()
    
    print("\nCheck the server terminal to see how the error handler processed these requests!")
    
    if not success:
        sys.exit(1) 
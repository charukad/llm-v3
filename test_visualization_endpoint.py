#!/usr/bin/env python3
"""
Test script for the NLP visualization endpoint with various visualization types.
"""

import requests
import json
import os
import argparse
from pprint import pprint
import time

def test_visualization(prompt, viz_type, server_url="http://localhost:8000"):
    """
    Test the NLP visualization endpoint with a given prompt and visualization type.
    
    Args:
        prompt: Natural language description of the visualization
        viz_type: Type of visualization (for informational purposes only)
        server_url: URL of the API server
    """
    # Endpoint for NLP visualization
    endpoint = f"{server_url}/nlp-visualization"
    
    # Request data
    request_data = {
        "prompt": prompt
    }
    
    print(f"Testing {viz_type} visualization")
    print(f"Endpoint: {endpoint}")
    print(f"Prompt: {prompt}")
    print("\nSending request...")
    
    # Make the request
    try:
        response = requests.post(endpoint, json=request_data, timeout=30)
        
        # Process the response
        if response.status_code == 200:
            result = response.json()
            
            print("\nResponse:")
            # Print nicely formatted JSON (without large base64 content)
            result_for_display = result.copy()
            if "base64_image" in result_for_display:
                b64_len = len(result_for_display["base64_image"])
                result_for_display["base64_image"] = f"[Base64 data: {b64_len} chars]"
            pprint(result_for_display)
            
            # Check if successful
            if result.get("success", False):
                print(f"\n{viz_type.capitalize()} visualization generated successfully!")
                if "file_path" in result:
                    file_path = result["file_path"]
                    print(f"File saved to: {file_path}")
                    
                    # Check if file exists
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"File size: {file_size} bytes")
                    else:
                        print("Warning: File not found locally (may be on server)")
                
                if "base64_image" in result:
                    base64_len = len(result["base64_image"])
                    print(f"Base64 image returned ({base64_len} characters)")
                
                return True
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                
                # If LLM analysis is available, print that too
                if "llm_analysis" in result and result["llm_analysis"]:
                    print("\nLLM Analysis:")
                    pprint(result["llm_analysis"])
                
                return False
        else:
            print(f"\nError: HTTP {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"\nException during request: {e}")
        return False

def test_all_visualization_types(server_url="http://localhost:8000"):
    """
    Test all supported visualization types.
    """
    # Define test cases for various visualization types
    test_cases = [
        {
            "name": "scatter plot",
            "prompt": """
            The dataset consists of 10 data points with corresponding x and y values. 
            The x-values range from 1 to 10, and the y-values are as follows: 
            when x is 1, y is 2; when x is 2, y is 3; when x is 3, y is 5; 
            when x is 4, y is 4; when x is 5, y is 6; when x is 6, y is 7; 
            when x is 7, y is 8; when x is 8, y is 7; when x is 9, y is 9; 
            and when x is 10, y is 10. Add a regression line.
            """
        },
        {
            "name": "2D function plot",
            "prompt": """
            Plot the function f(x) = sin(x) + cos(x) in the range of x from -2π to 2π.
            Label the x-axis as 'x' and the y-axis as 'f(x)'.
            Give it the title 'Sine plus Cosine Function'.
            """
        },
        {
            "name": "multiple 2D functions",
            "prompt": """
            Create a plot with three functions: f(x) = sin(x), g(x) = cos(x), and h(x) = sin(x) + cos(x).
            Use the range x from -π to π. Label each function in the legend as "Sine", "Cosine", and "Sum".
            Give the graph a title "Trigonometric Functions".
            """
        },
        {
            "name": "3D surface plot",
            "prompt": """
            Generate a 3D surface plot of the function f(x,y) = sin(sqrt(x^2 + y^2))/sqrt(x^2 + y^2),
            which is the 2D sinc function. Use the range -10 to 10 for both x and y.
            Title it "2D Sinc Function".
            """
        },
        {
            "name": "3D parametric curve",
            "prompt": """
            Create a 3D parametric plot of a helix with the following equations:
            x = cos(t), y = sin(t), z = t/2, where t ranges from 0 to 8π.
            Title it "Helix in 3D Space".
            """
        },
        {
            "name": "histogram",
            "prompt": """
            Create a histogram of a normal distribution with mean 0 and standard deviation 1.
            Generate 1000 random data points from this distribution.
            Use 30 bins and give it the title 'Normal Distribution Histogram'.
            """
        }
    ]
    
    # Run all tests
    results = {}
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*80}\nTest {i+1}/{len(test_cases)}: {test_case['name'].upper()}\n{'='*80}")
        
        # Run the test
        result = test_visualization(test_case["prompt"], test_case["name"], server_url)
        results[test_case["name"]] = result
        
        # Add a small delay between tests
        if i < len(test_cases) - 1:
            print("\nWaiting before next test...\n")
            time.sleep(2)
    
    # Print summary
    print(f"\n{'='*80}\nTEST SUMMARY\n{'='*80}")
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{name.ljust(25)}: {status}")
    
    # Return overall success
    return all(results.values())

def main():
    parser = argparse.ArgumentParser(description="Test NLP visualization endpoint")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                      help="Server URL")
    parser.add_argument("--type", type=str, choices=["scatter", "function", "functions", "3d", "parametric", "histogram", "all"],
                      default="all", help="Type of visualization to test")
    
    args = parser.parse_args()
    
    # If testing all types
    if args.type == "all":
        success = test_all_visualization_types(args.url)
        if success:
            print("\nAll visualization tests passed!")
        else:
            print("\nSome visualization tests failed.")
        return
    
    # Otherwise, test a specific type
    if args.type == "scatter":
        prompt = """
        The dataset consists of 10 data points with corresponding x and y values. 
        The x-values range from 1 to 10, and the y-values are as follows: 
        when x is 1, y is 2; when x is 2, y is 3; when x is 3, y is 5; 
        when x is 4, y is 4; when x is 5, y is 6; when x is 6, y is 7; 
        when x is 7, y is 8; when x is 8, y is 7; when x is 9, y is 9; 
        and when x is 10, y is 10. Add a regression line.
        """
    elif args.type == "function":
        prompt = """
        Plot the function f(x) = sin(x) + cos(x) in the range of x from -2π to 2π.
        Label the x-axis as 'x' and the y-axis as 'f(x)'.
        Give it the title 'Sine plus Cosine Function'.
        """
    elif args.type == "functions":
        prompt = """
        Create a plot with three functions: f(x) = sin(x), g(x) = cos(x), and h(x) = sin(x) + cos(x).
        Use the range x from -π to π. Label each function in the legend as "Sine", "Cosine", and "Sum".
        Give the graph a title "Trigonometric Functions".
        """
    elif args.type == "3d":
        prompt = """
        Generate a 3D surface plot of the function f(x,y) = sin(sqrt(x^2 + y^2))/sqrt(x^2 + y^2),
        which is the 2D sinc function. Use the range -10 to 10 for both x and y.
        Title it "2D Sinc Function".
        """
    elif args.type == "parametric":
        prompt = """
        Create a 3D parametric plot of a helix with the following equations:
        x = cos(t), y = sin(t), z = t/2, where t ranges from 0 to 8π.
        Title it "Helix in 3D Space".
        """
    elif args.type == "histogram":
        prompt = """
        Create a histogram of a normal distribution with mean 0 and standard deviation 1.
        Generate 1000 random data points from this distribution.
        Use 30 bins and give it the title 'Normal Distribution Histogram'.
        """
    
    # Run the test
    test_visualization(prompt, args.type, args.url)

if __name__ == "__main__":
    main() 
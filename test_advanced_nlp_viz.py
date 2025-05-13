#!/usr/bin/env python3
"""
Test script for advanced natural language visualization examples.
"""

import requests
import json
import os
import argparse
from pprint import pprint

def test_advanced_visualization(prompt, server_url="http://localhost:8000"):
    """Test the NLP visualization endpoint with advanced visualization examples."""
    # Endpoint for NLP visualization
    endpoint = f"{server_url}/nlp-visualization"
    
    # Request data
    request_data = {
        "prompt": prompt
    }
    
    print(f"Sending advanced NLP visualization request to {endpoint}")
    print(f"Prompt: {prompt}")
    print("\nSending request...")
    
    # Make the request
    response = requests.post(endpoint, json=request_data)
    
    # Process the response
    if response.status_code == 200:
        result = response.json()
        
        print("\nResponse:")
        # Print nicely formatted JSON, excluding base64 image for readability
        result_for_display = result.copy()
        if "base64_image" in result_for_display:
            result_for_display["base64_image"] = f"[Base64 data: {len(result_for_display['base64_image'])} chars]"
        pprint(result_for_display)
        
        # Check if successful
        if result.get("success", False):
            print("\nVisualization generated successfully!")
            if "file_path" in result:
                file_path = result["file_path"]
                print(f"File saved to: {file_path}")
                
                # Check if file exists
                if os.path.exists(file_path):
                    print(f"File size: {os.path.getsize(file_path)} bytes")
                else:
                    print("Warning: File not found locally (may be on server)")
            
            if "base64_image" in result:
                base64_len = len(result["base64_image"])
                print(f"Base64 image returned ({base64_len} characters)")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
            
            # If LLM analysis is available, print that too
            if "llm_analysis" in result and result["llm_analysis"]:
                print("\nLLM Analysis:")
                print(result["llm_analysis"])
    else:
        print(f"\nError: HTTP {response.status_code}")
        print(response.text)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test advanced NLP visualizations")
    parser.add_argument("--type", type=str, choices=["function", "multiple", "3d", "parametric", "scatter", "histogram"],
                      default="scatter", help="Type of visualization to test")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                      help="URL of the API server")
    args = parser.parse_args()
    
    # Set prompt based on visualization type
    if args.type == "function":
        prompt = """
        Plot the function f(x) = sin(x) + cos(x) in the range of x from -2π to 2π.
        Label the x-axis as 'x' and the y-axis as 'f(x) = sin(x) + cos(x)'.
        Give it the title 'Sine plus Cosine Function'.
        """
    elif args.type == "multiple":
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
    elif args.type == "scatter":
        prompt = """
        The dataset consists of 10 data points with corresponding x and y values. 
        The x-values range from 1 to 10, and the y-values are as follows: 
        when x is 1, y is 2; when x is 2, y is 3; when x is 3, y is 5; 
        when x is 4, y is 4; when x is 5, y is 6; when x is 6, y is 7; 
        when x is 7, y is 8; when x is 8, y is 7; when x is 9, y is 9; 
        and when x is 10, y is 10. These coordinate pairs can be used to plot 
        a basic scatterplot showing a general upward trend. Add a regression line.
        """
    elif args.type == "histogram":
        prompt = """
        Create a histogram of a normal distribution with mean 0 and standard deviation 1.
        Generate 1000 random data points from this distribution.
        Use 30 bins and give it the title "Normal Distribution Histogram".
        """
    
    # Run the test
    test_advanced_visualization(prompt, args.url)

if __name__ == "__main__":
    main() 
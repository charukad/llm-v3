#!/usr/bin/env python3
"""
Test script to demonstrate all the visualization types supported by the SuperVisualizationAgent.
This script tests various natural language prompts for different visualization types.
"""

import sys
import requests
import json
import os
import time
from datetime import datetime
import argparse

# List of test prompts for different visualization types
TEST_PROMPTS = [
    # Function 2D
    "Plot the function f(x) = sin(x) * cos(2*x) from -π to π",
    "Draw a graph of f(x) = x^3 - 2*x + 1",
    
    # Multiple functions 
    "Plot three functions on the same graph: sin(x), cos(x), and their sum sin(x)+cos(x)",
    
    # Function 3D
    "Create a 3D surface plot of z = sin(sqrt(x^2 + y^2))",
    "Generate a 3D visualization of the function f(x,y) = x^2 - y^2",
    
    # Histogram
    "Create a histogram of a normal distribution with mean 5 and standard deviation 2",
    "Show a histogram of 1000 random samples from an exponential distribution",
    
    # Scatter plot
    "Generate a scatter plot with these points: (1,3), (2,5), (3,4), (4,7), (5,8), (6,10) and show the trend line",
    "Plot a scatter diagram with linear regression line",
    
    # Boxplot
    "Create a boxplot comparing three groups: [1,2,3,4,5,6], [4,5,6,7,8], [2,3,4,5]",
    "Compare the distributions of two datasets with a boxplot",
    
    # Violin plot
    "Generate a violin plot for this dataset: [1,1,2,2,2,3,3,3,3,4,4,5]",
    "Compare distributions with a violin plot",
    
    # Bar chart
    "Create a bar chart with values [25, 40, 30, 55, 15] for categories [A, B, C, D, E]",
    "Make a horizontal bar graph showing sales data",
    
    # Pie chart
    "Create a pie chart showing market share: 30% for Apple, 25% for Samsung, 15% for Xiaomi and 30% for Others",
    "Generate a pie diagram of expenses",
    
    # Heatmap
    "Create a heatmap visualization of a correlation matrix",
    "Show data patterns with a heatmap",
    
    # Contour plot
    "Generate a contour plot of the function f(x,y) = x^2 + y^2",
    "Create a contour map of a bivariate function",
    
    # Complex function
    "Visualize the complex function f(z) = z^2 using domain coloring",
    "Show the phase portrait of a complex function",
    
    # Slope field
    "Plot the slope field for the differential equation dy/dx = y",
    "Show the direction field for a first-order ODE",
    
    # Time series
    "Generate a time series plot of stock prices over the last 30 days",
    "Visualize a time series with multiple seasonal patterns",
    
    # Correlation matrix
    "Create a correlation matrix visualization for 5 variables",
    "Show the relationships between variables with a correlation heatmap"
]

def test_visualization(prompt, server_url="http://localhost:8000"):
    """
    Test the NLP visualization endpoint with a given prompt.
    
    Args:
        prompt: Natural language description of the visualization
        server_url: URL of the API server
        
    Returns:
        Response from the server or None if there was an error
    """
    # Endpoint for NLP visualization
    endpoint = f"{server_url}/nlp-visualization"
    
    # Request data
    request_data = {
        "prompt": prompt
    }
    
    print(f"\n{'='*80}")
    print(f"Testing visualization prompt: {prompt}")
    print(f"{'='*80}")
    
    # Make the request
    try:
        response = requests.post(endpoint, json=request_data)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Success: {result.get('success')}")
            print(f"Visualization Type: {result.get('visualization_type')}")
            
            # Check if we have a file path
            if result.get('file_path'):
                print(f"File Path: {result.get('file_path')}")
                
                # Check if the file exists
                if os.path.exists(result.get('file_path')):
                    print(f"Visualization saved successfully!")
                else:
                    print(f"Warning: File not found at {result.get('file_path')}")
            
            # Check for errors
            if not result.get('success') and result.get('error'):
                print(f"Error: {result.get('error')}")
            
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
    parser = argparse.ArgumentParser(description="Test all visualization types supported by SuperVisualizationAgent")
    parser.add_argument("--url", type=str, help="Server URL", default="http://localhost:8000")
    parser.add_argument("--delay", type=float, help="Delay between tests in seconds", default=2.0)
    parser.add_argument("--prompt", type=str, help="Single test prompt (instead of running all tests)")
    parser.add_argument("--output", type=str, help="Output file to save test results")
    args = parser.parse_args()
    
    # Create results directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"viz_test_results_{timestamp}.txt")
    
    # Open output file
    with open(output_file, "w") as f:
        f.write(f"SuperVisualizationAgent Test Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Server URL: {args.url}\n\n")
        
        # Dictionary to store test results
        results = {}
        
        if args.prompt:
            # Test a single prompt
            print(f"Testing single prompt: {args.prompt}")
            f.write(f"Single prompt test: {args.prompt}\n")
            
            result = test_visualization(args.prompt, args.url)
            results[args.prompt] = result
            
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Success: {result.get('success') if result else False}\n")
            f.write(f"Visualization Type: {result.get('visualization_type') if result else 'N/A'}\n")
            f.write(f"File: {result.get('file_path') if result else 'N/A'}\n\n")
        else:
            # Test all prompts
            print(f"Testing {len(TEST_PROMPTS)} visualization prompts...")
            f.write(f"Testing {len(TEST_PROMPTS)} visualization prompts\n\n")
            
            success_count = 0
            
            for i, prompt in enumerate(TEST_PROMPTS):
                print(f"\nTest {i+1}/{len(TEST_PROMPTS)}")
                
                # Test the prompt
                result = test_visualization(prompt, args.url)
                results[prompt] = result
                
                # Write result to file
                f.write(f"Prompt {i+1}: {prompt}\n")
                f.write(f"Success: {result.get('success') if result else False}\n")
                f.write(f"Visualization Type: {result.get('visualization_type') if result else 'N/A'}\n")
                f.write(f"File: {result.get('file_path') if result else 'N/A'}\n\n")
                
                # Count successful tests
                if result and result.get('success'):
                    success_count += 1
                
                # Add delay between tests
                if i < len(TEST_PROMPTS) - 1:
                    time.sleep(args.delay)
            
            # Write summary
            success_rate = (success_count / len(TEST_PROMPTS)) * 100
            f.write(f"\nTest Summary:\n")
            f.write(f"Total tests: {len(TEST_PROMPTS)}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {len(TEST_PROMPTS) - success_count}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n")
            
            print(f"\nTest completed!")
            print(f"Results saved to: {output_file}")
            print(f"Success rate: {success_rate:.2f}% ({success_count}/{len(TEST_PROMPTS)})")

if __name__ == "__main__":
    main() 
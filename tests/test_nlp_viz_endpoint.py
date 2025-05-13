"""
Test the NLP visualization endpoint with realistic user queries.

This script sends a variety of realistic natural language prompts to the NLP visualization
endpoint and verifies that the responses are correct.
"""

import os
import sys
import json
import time
import requests
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set API endpoint
API_URL = "http://localhost:8000/nlp-visualization"
DEBUG_URL = "http://localhost:8000/nlp-visualization/debug"

def test_visualization_prompt(prompt: str, debug: bool = False) -> Dict[str, Any]:
    """
    Test a visualization prompt against the NLP visualization endpoint.
    
    Args:
        prompt: The natural language prompt to test
        debug: Whether to use the debug endpoint for more detailed information
        
    Returns:
        The response from the API
    """
    url = DEBUG_URL if debug else API_URL
    payload = {"prompt": prompt}
    
    print(f"\n==== Testing: {prompt} ====")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX errors
        
        result = response.json()
        
        if debug:
            # Print detailed debug information
            print(f"Detection method: {result.get('detection_method')}")
            print(f"Visualization type: {result.get('visualization_type')}")
            print(f"Pattern matching: {result.get('pattern_match_detected')}")
            print(f"LLM detection: {result.get('llm_detected')}")
        else:
            # Print basic response information
            print(f"Success: {result.get('success')}")
            print(f"Visualization type: {result.get('visualization_type')}")
            
            # Check if an image was generated
            if result.get('file_path'):
                print(f"Image saved to: {result.get('file_path')}")
            elif result.get('base64_image'):
                print(f"Base64 image returned (length: {len(result.get('base64_image'))})")
            
            # Check for error message
            if result.get('error'):
                print(f"Error: {result.get('error')}")
        
        return result
    
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")
        return {"success": False, "error": str(e)}

def run_realistic_tests():
    """Run a series of realistic user queries to test the endpoint."""
    
    # Define realistic prompts grouped by visualization type
    realistic_prompts = [
        # 2D Functions
        "Draw me a graph of y = x^2 - 3x + 2",
        "I need to visualize f(x) = sin(x) / x from -10 to 10",
        "Can you plot the exponential function e^(-x^2) for me?",
        "Show the graph of a parabola y = x^2",
        "Graph the function f(x) = log(x) for x > 0",
        
        # 3D Functions
        "Create a 3D plot of z = sin(x) * cos(y)",
        "I need a 3D visualization of the function f(x,y) = x^2 + y^2",
        "Generate a 3D surface for z = x*e^(-x^2-y^2)",
        "Show me a 3D graph of a simple hill z = e^(-(x^2+y^2))",
        "Make a 3D surface plot for z = sin(sqrt(x^2 + y^2))",
        
        # Statistical plots
        "Create a histogram of a normal distribution with mean 5 and std 2",
        "Generate a boxplot comparing the following datasets: [1,2,3,4,5], [2,4,6,8,10], [3,6,9,12,15]",
        "I need a scatter plot of these points: (1,3), (2,5), (4,4), (5,7), (8,9)",
        "Create a pie chart showing market share with values 30%, 25%, 15%, 20%, and 10%",
        "Make a violin plot comparing these distributions: [1,2,2,3,3,3,4,4,5], [2,3,3,4,4,4,5,5,6]",
        
        # Advanced plots
        "Generate a contour plot of f(x,y) = sin(x) + cos(y)",
        "Visualize the complex function f(z) = z^2",
        "Create a correlation matrix for 5 variables",
        "Show a slope field for the differential equation y' = y",
        "Create a time series of monthly temperatures over a year",
        
        # Ambiguous/challenging queries
        "Show me some data visualization",
        "Plot a mathematical function",
        "Create a pretty graph",
        "Can you visualize sin(x) and cos(x) together?",
        "I need to see how x^2 and x^3 compare"
    ]
    
    results = []
    
    # Test each prompt and collect results
    for prompt in realistic_prompts:
        # Test with both regular and debug endpoints
        result = test_visualization_prompt(prompt)
        debug_result = test_visualization_prompt(prompt, debug=True)
        
        # Store the results for analysis
        results.append({
            "prompt": prompt,
            "success": result.get("success", False),
            "visualization_type": result.get("visualization_type"),
            "detection_method": debug_result.get("detection_method"),
            "error": result.get("error")
        })
        
        # Small delay to avoid overwhelming the server
        time.sleep(1)
    
    # Analyze results
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    # Group results by visualization type
    vis_types = {}
    for r in results:
        vis_type = r.get("visualization_type", "unknown")
        if vis_type not in vis_types:
            vis_types[vis_type] = {"count": 0, "success": 0}
        vis_types[vis_type]["count"] += 1
        if r["success"]:
            vis_types[vis_type]["success"] += 1
    
    # Print summary
    print("\n===== TEST RESULTS =====")
    print(f"Overall success rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    print("\nResults by visualization type:")
    for vis_type, stats in vis_types.items():
        success_rate = (stats["success"] / stats["count"]) * 100
        print(f"{vis_type}: {stats['success']}/{stats['count']} ({success_rate:.1f}%)")
    
    # Show detailed results for failures
    print("\nFailed queries:")
    for r in results:
        if not r["success"]:
            print(f"- {r['prompt']} (Type: {r['visualization_type']}, Error: {r['error']})")

    return results

if __name__ == "__main__":
    run_realistic_tests() 
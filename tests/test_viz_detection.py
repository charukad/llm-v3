"""
Test the enhanced visualization detection for different types of plots.

This script tests various natural language prompts against the pattern-matching-based
visualization detector to verify that the correct visualization types are detected.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the extract_visualization_parameters function
from api.rest.routes.nlp_visualization import extract_visualization_parameters

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prompt(prompt: str) -> Dict[str, Any]:
    """
    Test a visualization prompt and return the detected parameters.
    
    Args:
        prompt: The natural language prompt to test
        
    Returns:
        The detected visualization parameters
    """
    logger.info(f"Testing prompt: {prompt}")
    result = extract_visualization_parameters(prompt)
    
    if result:
        logger.info(f"✓ Detected visualization type: {result.get('visualization_type')}")
        # Print a subset of parameters for readability
        if 'parameters' in result:
            params = result['parameters']
            param_subset = {k: v for k, v in params.items() 
                           if k not in ['data'] or isinstance(v, (str, int, float, bool))}
            logger.info(f"Parameters: {json.dumps(param_subset, indent=2)}")
        return result
    else:
        logger.warning(f"✗ No visualization type detected for: {prompt}")
        return {}

def main():
    """Run tests for various visualization types."""
    
    # Series of test prompts organized by visualization type
    test_prompts = {
        "3D Surface": [
            "Create a 3D surface plot of z = sin(x) * cos(y)",
            "Plot f(x,y) = x^2 + y^2 as a 3D surface",
            "Show a 3D graph of the function z = exp(-(x^2+y^2))",
            "3D visualization of x^2 - y^2",
            "Make a 3D surface plot"
        ],
        "2D Function": [
            "Plot f(x) = sin(x) * cos(2*x) from -π to π",
            "Graph y = x^2 - 4",
            "Draw the function f(x) = exp(-x^2)",
            "Plot sinx + cosx from -10 to 10",
            "Visualize the curve y = log(x)"
        ],
        "Contour": [
            "Create a contour plot of f(x,y) = x^2 + y^2",
            "Show level curves for z = sin(x)*cos(y)",
            "Make a filled contour plot of x^2 - y^2",
            "Visualize the equipotential lines of function z = 1/sqrt(x^2 + y^2)",
            "Contour plot with 15 levels"
        ],
        "Complex Function": [
            "Visualize the complex function f(z) = z^2",
            "Domain coloring for f(z) = 1/z",
            "Plot the phase of the complex function sin(z)",
            "Visualize the absolute value of e^z in the complex plane",
            "Complex function plot of z^3 - 1"
        ],
        "Time Series": [
            "Create a time series plot of stock prices",
            "Plot the temperature data over time",
            "Visualize this time series: [1, 2, 3, 5, 8, 13, 21]",
            "Show the temporal evolution of the data",
            "Create a time-dependent visualization of data"
        ],
        "Correlation Matrix": [
            "Create a correlation matrix for my variables",
            "Visualize the correlation between 5 variables",
            "Show a heatmap of correlation coefficients",
            "Plot the correlation matrix for my dataset",
            "Correlation heatmap visualization"
        ],
        "Slope Field": [
            "Create a slope field for dy/dx = y",
            "Vector field plot for the differential equation y' = sin(x)",
            "Visualize the direction field for dy/dx = x*y",
            "Slope field with solution curves for y' = -y",
            "Direction field for the ODE y' = x^2"
        ],
        "Other Types": [
            "Create a histogram of normal distribution with mean 0 and std 1",
            "Make a scatter plot of these points: (1,2), (3,4), (5,6)",
            "Visualize a box plot of three datasets",
            "Create a pie chart with values 30, 20, 15, 10, 25",
            "Violin plot comparison of these datasets"
        ]
    }
    
    results = {}
    
    # Run all tests
    for category, prompts in test_prompts.items():
        logger.info(f"\n===== TESTING {category} PROMPTS =====")
        
        category_results = []
        for prompt in prompts:
            result = test_prompt(prompt)
            category_results.append({
                "prompt": prompt,
                "detected": bool(result),
                "visualization_type": result.get("visualization_type", None)
            })
        
        results[category] = category_results
    
    # Calculate success rates by category
    logger.info("\n===== TEST RESULTS =====")
    overall_success = 0
    overall_total = 0
    
    for category, category_results in results.items():
        success = sum(1 for r in category_results if r["detected"])
        total = len(category_results)
        success_rate = (success / total) * 100 if total > 0 else 0
        
        logger.info(f"{category}: {success}/{total} ({success_rate:.1f}%)")
        
        overall_success += success
        overall_total += total
    
    overall_rate = (overall_success / overall_total) * 100 if overall_total > 0 else 0
    logger.info(f"\nOverall Success Rate: {overall_success}/{overall_total} ({overall_rate:.1f}%)")
    
    return results

if __name__ == "__main__":
    main() 
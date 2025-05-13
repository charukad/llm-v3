import requests
import json
import sys
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoint
API_ENDPOINT = "http://localhost:8000/api/nlp-visualization/debug"

def test_visualization_detection(prompt: str) -> Dict[str, Any]:
    """
    Test the visualization detection for a given prompt.
    
    Args:
        prompt: The natural language prompt to test
        
    Returns:
        The debug response from the API
    """
    try:
        response = requests.post(
            API_ENDPOINT,
            json={"prompt": prompt},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"API Error: {response.status_code}"
            }
            
    except Exception as e:
        logger.exception(f"Error testing visualization: {e}")
        return {
            "success": False,
            "error": f"Request Error: {str(e)}"
        }

def analyze_detection_result(result: Dict[str, Any], expected_type: str) -> bool:
    """
    Analyze the detection result and check if it matches the expected type.
    
    Args:
        result: The detection result
        expected_type: The expected visualization type
        
    Returns:
        True if the detection was correct, False otherwise
    """
    detected_type = result.get("visualization_type")
    detection_method = result.get("detection_method", "None")
    
    if detected_type == expected_type:
        logger.info(f"✅ SUCCESS: Detected {detected_type} using {detection_method}")
        return True
    else:
        logger.error(f"❌ FAILED: Expected {expected_type}, but got {detected_type} using {detection_method}")
        
        # Show detailed debug information
        if result.get("pattern_match_logs"):
            logger.debug("Pattern match logs:")
            for log in result.get("pattern_match_logs", []):
                logger.debug(f"  {log.get('level')}: {log.get('message')}")
                
        if result.get("llm_logs"):
            logger.debug("LLM logs:")
            for log in result.get("llm_logs", []):
                logger.debug(f"  {log.get('level')}: {log.get('message')}")
        
        return False

def run_test_suite(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run a suite of test cases.
    
    Args:
        test_cases: List of test cases, each with a prompt and expected type
        
    Returns:
        Test summary statistics
    """
    results = {
        "total": len(test_cases),
        "success": 0,
        "failure": 0,
        "by_type": {}
    }
    
    for i, test_case in enumerate(test_cases):
        prompt = test_case["prompt"]
        expected_type = test_case["expected_type"]
        
        logger.info(f"\nTest {i+1}/{len(test_cases)}: {expected_type}")
        logger.info(f"Prompt: {prompt}")
        
        result = test_visualization_detection(prompt)
        success = analyze_detection_result(result, expected_type)
        
        # Update statistics
        if success:
            results["success"] += 1
        else:
            results["failure"] += 1
            
        # Update per-type statistics
        if expected_type not in results["by_type"]:
            results["by_type"][expected_type] = {
                "total": 0,
                "success": 0,
                "failure": 0
            }
            
        results["by_type"][expected_type]["total"] += 1
        if success:
            results["by_type"][expected_type]["success"] += 1
        else:
            results["by_type"][expected_type]["failure"] += 1
    
    return results

def print_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the test results.
    
    Args:
        results: Test summary statistics
    """
    logger.info("\n" + "=" * 50)
    logger.info(f"TEST SUMMARY: {results['success']}/{results['total']} passed ({results['success']/results['total']*100:.1f}%)")
    logger.info("=" * 50)
    
    logger.info("\nResults by visualization type:")
    for viz_type, stats in results["by_type"].items():
        success_rate = stats["success"] / stats["total"] * 100
        logger.info(f"{viz_type}: {stats['success']}/{stats['total']} passed ({success_rate:.1f}%)")
    
    logger.info("\n" + "=" * 50)
    
    # Identify problem areas
    problem_types = []
    for viz_type, stats in results["by_type"].items():
        if stats["success"] < stats["total"]:
            problem_types.append((viz_type, stats["success"] / stats["total"] * 100))
    
    if problem_types:
        logger.info("\nProblem areas (in order of priority):")
        problem_types.sort(key=lambda x: x[1])
        for viz_type, success_rate in problem_types:
            logger.info(f"- {viz_type}: {success_rate:.1f}% success rate")

def main():
    """Main test function."""
    # Define test cases
    test_cases = [
        # 3D Surface Plots - Common problem area
        {"prompt": "Create a 3D surface plot of z = sin(x)*cos(y)", "expected_type": "function_3d"},
        {"prompt": "Plot the function f(x,y) = x^2 + y^2 in 3D", "expected_type": "function_3d"},
        {"prompt": "Make a 3D visualization of the function x^2 - y^2", "expected_type": "function_3d"},
        {"prompt": "3D surface plot for z = e^(-(x^2+y^2))", "expected_type": "function_3d"},
        {"prompt": "Generate a 3D surface showing z = sin(sqrt(x^2 + y^2))", "expected_type": "function_3d"},
        
        # 2D Function Plots
        {"prompt": "Plot y = sin(x) from -π to π", "expected_type": "function_2d"},
        {"prompt": "Create a graph of f(x) = x^2 - 2*x + 1", "expected_type": "function_2d"},
        {"prompt": "Plot the function log(x) over the range [1, 10]", "expected_type": "function_2d"},
        {"prompt": "Graph of y = tan(x) in the interval [-1.5, 1.5]", "expected_type": "function_2d"},
        
        # Contour Plots
        {"prompt": "Create a contour plot of f(x,y) = x^2 + y^2", "expected_type": "contour"},
        {"prompt": "Generate 15 level curves for z = sin(x)*cos(y)", "expected_type": "contour"},
        {"prompt": "Make a filled contour plot of z = x^2 - y^2", "expected_type": "contour"},
        {"prompt": "Show equipotential lines for f(x,y) = log(x^2 + y^2)", "expected_type": "contour"},
        
        # Complex Functions
        {"prompt": "Plot the complex function f(z) = z^2", "expected_type": "complex_function"},
        {"prompt": "Create a domain coloring for f(z) = 1/z", "expected_type": "complex_function"},
        {"prompt": "Visualize the complex function e^z", "expected_type": "complex_function"},
        {"prompt": "Phase plot of f(z) = sin(z)", "expected_type": "complex_function"},
        {"prompt": "Show the absolute value of the complex function z^3 - 1", "expected_type": "complex_function"},
        
        # Time Series
        {"prompt": "Create a time series plot of temperature over time", "expected_type": "time_series"},
        {"prompt": "Plot the stock price data over the last 100 days", "expected_type": "time_series"},
        {"prompt": "Time series visualization of seasonal temperature changes", "expected_type": "time_series"},
        {"prompt": "Show temporal evolution of data [1, 2, 3, 5, 8, 13, 21]", "expected_type": "time_series"},
        
        # Correlation Matrix
        {"prompt": "Create a correlation matrix visualization", "expected_type": "correlation_matrix"},
        {"prompt": "Generate a heatmap showing correlations between variables", "expected_type": "correlation_matrix"},
        {"prompt": "Show a correlation heatmap for my dataset", "expected_type": "correlation_matrix"},
        
        # Slope Fields
        {"prompt": "Create a slope field for dy/dx = y", "expected_type": "slope_field"},
        {"prompt": "Generate a direction field for the differential equation y' = sin(x) + y", "expected_type": "slope_field"},
        {"prompt": "Visualize the slope field with solution curves for dy/dx = x^2 - y", "expected_type": "slope_field"},
        
        # Histograms
        {"prompt": "Create a histogram of a normal distribution with mean 0 and std 1", "expected_type": "histogram"},
        {"prompt": "Generate a histogram with 20 bins for a uniform distribution", "expected_type": "histogram"},
        {"prompt": "Show a frequency distribution with density curve", "expected_type": "histogram"},
        
        # Box Plots
        {"prompt": "Create a box plot comparing three datasets", "expected_type": "boxplot"},
        {"prompt": "Generate a horizontal box and whisker plot", "expected_type": "boxplot"},
        {"prompt": "Visualize the distribution of data using box plots", "expected_type": "boxplot"},
        
        # Pie Charts
        {"prompt": "Create a pie chart with categories A (30%), B (45%), C (25%)", "expected_type": "pie"},
        {"prompt": "Generate a pie graph showing market share distribution", "expected_type": "pie"},
        {"prompt": "Make a circle graph of budget allocation", "expected_type": "pie"},
        
        # Scatter Plots
        {"prompt": "Create a scatter plot of points (1,2), (3,4), (5,6)", "expected_type": "scatter"},
        {"prompt": "Generate a scatter plot with regression line", "expected_type": "scatter"},
        {"prompt": "Plot data points showing correlation between x and y", "expected_type": "scatter"},
        
        # Violin Plots
        {"prompt": "Create a violin plot of my datasets", "expected_type": "violin"},
        {"prompt": "Compare distributions using violin plots", "expected_type": "violin"},
        {"prompt": "Make horizontal violin charts for the data", "expected_type": "violin"},
        
        # Edge cases and confusing cases
        {"prompt": "Plot a 3D function that looks like a saddle z = x^2 - y^2", "expected_type": "function_3d"},
        {"prompt": "Create both contour and 3D surface for z = sin(x)*cos(y)", "expected_type": "function_3d"},
        {"prompt": "First make a 3D surface showing f(x,y) = x^2 + y^2", "expected_type": "function_3d"},
        {"prompt": "Visualize f(x,y) = cos(x) + sin(y) both as a contour and a surface", "expected_type": "function_3d"}
    ]
    
    # Run the tests
    results = run_test_suite(test_cases)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main() 
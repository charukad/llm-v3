import requests
import json
import logging
from typing import Dict, Any, List, Tuple
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoint
API_ENDPOINT = "http://localhost:8000/api/nlp-visualization"

def test_visualization_api(prompt: str) -> Dict[str, Any]:
    """
    Test the NLP visualization API with a prompt.
    
    Args:
        prompt: The natural language prompt to test
        
    Returns:
        The response from the API
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

def test_debug_api(prompt: str) -> Dict[str, Any]:
    """
    Test the debug endpoint for visualization detection.
    
    Args:
        prompt: The natural language prompt to test
        
    Returns:
        Debug information from the API
    """
    try:
        response = requests.post(
            API_ENDPOINT + "/debug",
            json={"prompt": prompt},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Debug API Error: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"Debug API Error: {response.status_code}"
            }
            
    except Exception as e:
        logger.exception(f"Error testing debug endpoint: {e}")
        return {
            "success": False,
            "error": f"Request Error: {str(e)}"
        }

def run_test_cases() -> None:
    """Run specific test cases focusing on the problem areas."""
    
    # Test cases focusing on the main problem areas
    test_cases = [
        # 3D Surface plots
        ("Create a 3D surface plot of z = sin(x)*cos(y)", "function_3d", "3D Surface Detection"),
        ("Plot the function f(x,y) = x^2 + y^2 in 3D", "function_3d", "3D Function with f(x,y) notation"),
        ("Make a 3D visualization of the function x^2 - y^2", "function_3d", "3D Function without explicit z="),
        ("Generate a surface plot showing z = sin(sqrt(x^2 + y^2))", "function_3d", "3D Surface with complex expression"),
        
        # Contour plots
        ("Create a contour plot of f(x,y) = x^2 + y^2", "contour", "Basic contour plot"),
        ("Generate level curves for z = sin(x)*cos(y)", "contour", "Contour with level curves terminology"),
        ("Show equipotential lines for f(x,y) = log(x^2 + y^2)", "contour", "Contour with equipotential terminology"),
        
        # Complex functions
        ("Plot the complex function f(z) = z^2", "complex_function", "Basic complex function"),
        ("Create a domain coloring for f(z) = 1/z", "complex_function", "Complex with domain coloring"),
        ("Visualize the complex function e^z", "complex_function", "Complex with e^z notation"),
        
        # Edge cases with potential confusion
        ("Plot a 3D function that looks like a saddle z = x^2 - y^2", "function_3d", "3D with saddle description"),
        ("Show both a contour and a 3D surface for z = sin(x)*cos(y)", "function_3d", "3D with contour mention"),
        ("First create a contour plot and then a 3D surface of z = x^2 + y^2", "contour", "Mixed request starting with contour")
    ]
    
    # Test results
    results = {
        "total": len(test_cases),
        "success": 0,
        "failure": 0,
        "details": []
    }
    
    for i, (prompt, expected_type, description) in enumerate(test_cases):
        logger.info(f"\nTest {i+1}/{len(test_cases)}: {description}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Expected type: {expected_type}")
        
        # First get detailed debug information
        debug_result = test_debug_api(prompt)
        detected_type = debug_result.get("visualization_type")
        detection_method = debug_result.get("detection_method", "unknown")
        
        logger.info(f"Detection method: {detection_method}")
        logger.info(f"Detected type: {detected_type}")
        
        # Track success
        success = detected_type == expected_type
        if success:
            logger.info(f"✅ SUCCESS: Correctly detected {detected_type} using {detection_method}")
            results["success"] += 1
        else:
            logger.error(f"❌ FAILED: Expected {expected_type}, but got {detected_type} using {detection_method}")
            results["failure"] += 1
        
        # Record detailed results
        results["details"].append({
            "prompt": prompt,
            "description": description,
            "expected_type": expected_type,
            "detected_type": detected_type,
            "detection_method": detection_method,
            "success": success
        })
        
        # Add a small delay between tests to avoid overwhelming the API
        time.sleep(1)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info(f"TEST SUMMARY: {results['success']}/{results['total']} passed ({results['success']/results['total']*100:.1f}%)")
    logger.info("=" * 50)
    
    return results

def main():
    """Main function to run the tests."""
    
    logger.info("Testing improved visualization detection...")
    results = run_test_cases()
    
    # Check if we achieved 100% success
    if results["success"] == results["total"]:
        logger.info("🎉 All tests passed! The improvements were successful.")
    else:
        logger.warning(f"Some tests failed: {results['failure']}/{results['total']} failures.")
        
        # Show details of failures
        logger.info("\nFailed tests:")
        for item in results["details"]:
            if not item["success"]:
                logger.info(f"- {item['description']}: Expected {item['expected_type']}, got {item['detected_type']}")

if __name__ == "__main__":
    main() 
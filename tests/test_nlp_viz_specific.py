"""
Test specific visualization types with focused queries.

This script allows testing specific visualization types to verify
that our enhancements are working correctly for priority use cases.
"""

import os
import sys
import json
import time
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure API endpoints
API_URL = "http://localhost:8000/nlp-visualization"
DEBUG_URL = "http://localhost:8000/nlp-visualization/debug"

class VisualizationTester:
    """Tester for visualization endpoints."""
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Initialize the tester.
        
        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def test_prompt(self, prompt: str, save_image: bool = True, show_debug: bool = False) -> Dict[str, Any]:
        """
        Test a single prompt and save/display the result.
        
        Args:
            prompt: The natural language prompt to test
            save_image: Whether to save the generated image
            show_debug: Whether to show debug information
            
        Returns:
            The API response data
        """
        print(f"\nTesting: {prompt}")
        
        if show_debug:
            return self._test_debug(prompt)
        else:
            return self._test_visualization(prompt, save_image)
    
    def _test_visualization(self, prompt: str, save_image: bool) -> Dict[str, Any]:
        """Test the main visualization endpoint."""
        try:
            response = requests.post(API_URL, json={"prompt": prompt})
            response.raise_for_status()
            
            result = response.json()
            
            # Print basic result
            print(f"Success: {result.get('success', False)}")
            print(f"Visualization type: {result.get('visualization_type')}")
            
            # Save or show image if available
            if result.get("success", False):
                if result.get("file_path"):
                    print(f"Image saved at: {result.get('file_path')}")
                elif result.get("base64_image") and save_image:
                    self._save_base64_image(result["base64_image"], prompt, result.get("visualization_type"))
            elif result.get("error"):
                print(f"Error: {result.get('error')}")
            
            return result
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _test_debug(self, prompt: str) -> Dict[str, Any]:
        """Test the debug endpoint."""
        try:
            response = requests.post(DEBUG_URL, json={"prompt": prompt})
            response.raise_for_status()
            
            result = response.json()
            
            # Print detailed debug info
            print(f"Detection method: {result.get('detection_method')}")
            print(f"Visualization type: {result.get('visualization_type')}")
            print(f"Pattern matching detected: {result.get('pattern_match_detected')}")
            
            if result.get("pattern_match_result"):
                print(f"Pattern match type: {result.get('pattern_match_result', {}).get('visualization_type')}")
            
            # Print some pattern match logs if available
            logs = result.get("pattern_match_logs", [])
            if logs:
                print("\nPattern matching logs:")
                for log in logs[:5]:  # Show first 5 logs
                    print(f"- {log.get('message')}")
                
            return result
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _save_base64_image(self, base64_str: str, prompt: str, vis_type: str):
        """Save a base64 encoded image to file."""
        try:
            # Create a safe filename
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
            filename = f"{vis_type}_{safe_prompt}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Decode and save image
            img_data = base64.b64decode(base64_str)
            with open(filepath, "wb") as f:
                f.write(img_data)
            
            print(f"Image saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
    
    def test_3d_functions(self):
        """Test 3D function visualizations."""
        print("\n===== TESTING 3D FUNCTIONS =====")
        prompts = [
            "Create a 3D plot of z = sin(x) * cos(y)",
            "Create a 3D surface plot of z = sin(x) * cos(y)",
            "Generate a 3D surface for z = x*e^(-x^2-y^2)",
            "I need a 3D visualization of the function f(x,y) = x^2 + y^2",
            "Show me a 3D surface of f(x,y) = sin(sqrt(x^2 + y^2))",
            "Plot the 3D function z = x^2 - y^2",
            "Create a 3D surface plot showing a saddle shape z = x^2 - y^2",
            "Show me a 3D mountain plot where z = e^(-(x^2+y^2))"
        ]
        
        results = []
        for prompt in prompts:
            result = self.test_prompt(prompt)
            results.append({
                "prompt": prompt,
                "success": result.get("success", False),
                "visualization_type": result.get("visualization_type"),
                "error": result.get("error")
            })
            time.sleep(0.5)  # Small delay
        
        return results
    
    def test_statistical_plots(self):
        """Test statistical visualization types."""
        print("\n===== TESTING STATISTICAL PLOTS =====")
        prompts = [
            "Create a histogram of a normal distribution with mean 5 and std 2",
            "Generate a boxplot comparing these datasets: [1,2,3,4,5], [2,4,6,8,10]",
            "I need a scatter plot of these points: (1,3), (2,5), (4,4), (5,7), (8,9)",
            "Show me a pie chart with values 30, 25, 15, 20, 10",
            "Create a violin plot comparison of [1,2,3,3,3,4,5] and [2,3,4,4,5,6,7]",
            "Make a bar chart showing sales by region: North 120, South 85, East 95, West 110",
            "Create a heatmap of the following data matrix: [[1,2,3],[4,5,6],[7,8,9]]",
            "Generate a correlation matrix visualization"
        ]
        
        results = []
        for prompt in prompts:
            result = self.test_prompt(prompt)
            results.append({
                "prompt": prompt,
                "success": result.get("success", False),
                "visualization_type": result.get("visualization_type"),
                "error": result.get("error")
            })
            time.sleep(0.5)  # Small delay
        
        return results
    
    def test_advanced_plots(self):
        """Test advanced visualization types."""
        print("\n===== TESTING ADVANCED PLOTS =====")
        prompts = [
            "Generate a contour plot of f(x,y) = sin(x) + cos(y)",
            "Create a filled contour plot of z = x^2 + y^2",
            "Visualize the complex function f(z) = z^2",
            "Show the phase plot of the complex function f(z) = 1/z",
            "Create a correlation matrix for 5 variables",
            "Show a slope field for the differential equation y' = y",
            "Direction field for the ODE dy/dx = x - y",
            "Create a time series of monthly temperatures over a year",
            "Plot a time series of stock prices over 100 days"
        ]
        
        results = []
        for prompt in prompts:
            result = self.test_prompt(prompt)
            results.append({
                "prompt": prompt,
                "success": result.get("success", False),
                "visualization_type": result.get("visualization_type"),
                "error": result.get("error")
            })
            time.sleep(0.5)  # Small delay
        
        return results
    
    def test_all(self):
        """Run all tests and summarize results."""
        all_results = []
        
        # Test each category
        all_results.extend(self.test_3d_functions())
        all_results.extend(self.test_statistical_plots())
        all_results.extend(self.test_advanced_plots())
        
        # Calculate success rate
        success_count = sum(1 for r in all_results if r["success"])
        total_count = len(all_results)
        success_rate = (success_count / total_count) * 100
        
        # Group by visualization type
        vis_types = {}
        for r in all_results:
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
            type_success_rate = (stats["success"] / stats["count"]) * 100
            print(f"{vis_type}: {stats['success']}/{stats['count']} ({type_success_rate:.1f}%)")
        
        # Show detailed results for failures
        failures = [r for r in all_results if not r["success"]]
        if failures:
            print("\nFailed queries:")
            for r in failures:
                print(f"- {r['prompt']} (Type: {r['visualization_type']}, Error: {r['error']})")
        
        return all_results

if __name__ == "__main__":
    tester = VisualizationTester()
    
    # Select which tests to run
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "3d":
            tester.test_3d_functions()
        elif test_type == "stats":
            tester.test_statistical_plots()
        elif test_type == "advanced":
            tester.test_advanced_plots()
        else:
            print(f"Unknown test type: {test_type}")
    else:
        # Run all tests
        tester.test_all() 
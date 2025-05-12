#!/usr/bin/env python3

"""
Script to test the advanced visualization agent capabilities.
"""

import os
import sys
import json

# Add parent directory to Python path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent

def main():
    # Initialize the advanced visualization agent
    agent = AdvancedVisualizationAgent({
        "storage_dir": "output_visualizations",
        "use_database": False,
        "default_format": "png"
    })
    
    # Create a temporary output directory
    os.makedirs("output_visualizations", exist_ok=True)
    
    # Test 1: Function with derivative
    test_derivative_message = {
        "header": {
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "derivative",
            "parameters": {
                "expression": "sin(x) * cos(x)",
                "variable": "x",
                "x_range": (-10, 10),
                "save": True
            }
        }
    }
    
    # Test 2: Function with critical points
    test_critical_points_message = {
        "header": {
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "critical_points",
            "parameters": {
                "expression": "sin(x) * cos(x)",
                "variable": "x",
                "x_range": (-10, 10),
                "save": True
            }
        }
    }
    
    # Process derivative message
    print("Processing derivative visualization request...")
    derivative_result = agent.process_message(test_derivative_message)
    
    if derivative_result.get('success'):
        print(f"Derivative visualization saved to: {derivative_result.get('file_path')}")
        # Open the file
        if sys.platform == 'darwin':  # macOS
            os.system(f"open {derivative_result['file_path']}")
    else:
        print(f"Error with derivative visualization: {derivative_result.get('error')}")
    
    # Process critical points message
    print("\nProcessing critical points visualization request...")
    critical_points_result = agent.process_message(test_critical_points_message)
    
    if critical_points_result.get('success'):
        print(f"Critical points visualization saved to: {critical_points_result.get('file_path')}")
        # Open the file
        if sys.platform == 'darwin':  # macOS
            os.system(f"open {critical_points_result['file_path']}")
    else:
        print(f"Error with critical points visualization: {critical_points_result.get('error')}")

if __name__ == "__main__":
    main() 
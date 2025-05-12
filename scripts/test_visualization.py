#!/usr/bin/env python3
"""
Test script for the Visualization component.

This script tests the basic functionality of the Visualization component
by generating various types of visualizations and verifying the results.
"""

import os
import sys
import json
import sympy as sp

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.agent.viz_agent import VisualizationAgent
from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent
from visualization.selection.context_analyzer import VisualizationSelector

def test_2d_function_plot():
    """Test 2D function plotting."""
    print("\n=== Testing 2D Function Plot ===")
    agent = VisualizationAgent({"storage_dir": "test_visualizations", "use_database": False})
    
    message = {
        "header": {
            "message_id": "test_2d",
            "sender": "test",
            "recipient": "visualization_agent",
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "function_2d",
            "parameters": {
                "expression": "sin(x) + 0.5*sin(3*x)",
                "x_range": (-10, 10),
                "title": "Test 2D Function Plot"
            }
        }
    }
    
    result = agent.process_message(message)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if result.get("success", False):
        print(f"Visualization saved to: {result.get('file_path')}")
    else:
        print(f"Error: {result.get('error')}")

def test_3d_function_plot():
    """Test 3D function plotting."""
    print("\n=== Testing 3D Function Plot ===")
    agent = VisualizationAgent({"storage_dir": "test_visualizations", "use_database": False})
    
    message = {
        "header": {
            "message_id": "test_3d",
            "sender": "test",
            "recipient": "visualization_agent",
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "function_3d",
            "parameters": {
                "expression": "sin(sqrt(x**2 + y**2))",
                "x_range": (-5, 5),
                "y_range": (-5, 5),
                "title": "Test 3D Function Plot"
            }
        }
    }
    
    result = agent.process_message(message)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if result.get("success", False):
        print(f"Visualization saved to: {result.get('file_path')}")
    else:
        print(f"Error: {result.get('error')}")

def test_derivative_plot():
    """Test derivative plotting."""
    print("\n=== Testing Derivative Plot ===")
    agent = AdvancedVisualizationAgent({"storage_dir": "test_visualizations", "use_database": False})
    
    message = {
        "header": {
            "message_id": "test_derivative",
            "sender": "test",
            "recipient": "visualization_agent",
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "derivative",
            "parameters": {
                "expression": "x**3 - 3*x**2 + 2*x - 1",
                "variable": "x",
                "x_range": (-2, 4),
                "title": "Function and Derivative"
            }
        }
    }
    
    result = agent.process_message(message)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if result.get("success", False):
        print(f"Visualization saved to: {result.get('file_path')}")
    else:
        print(f"Error: {result.get('error')}")

def test_visualization_selector():
    """Test visualization selection logic."""
    print("\n=== Testing Visualization Selector ===")
    selector = VisualizationSelector()
    
    # Test contexts
    contexts = [
        {
            "expression": "x**2 - 4*x + 4",
            "domain": "algebra",
            "operation": "solve"
        },
        {
            "expression": "sin(x)",
            "domain": "calculus",
            "operation": "differentiate"
        },
        {
            "expression": "x**2 + y**2",
            "domain": "calculus"
        },
        {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "data_type": "histogram"
        },
        {
            "x_component": "y",
            "y_component": "-x",
            "operation": "vector field"
        }
    ]
    
    for i, context in enumerate(contexts):
        print(f"\nContext {i+1}: {context}")
        result = selector.select_visualization(context)
        print(f"Recommended: {result.get('recommended_visualization', {}).get('type')}")
        print(f"Parameters: {json.dumps(result.get('recommended_visualization', {}).get('params', {}), indent=2)}")
        if "alternative_visualizations" in result:
            print(f"Alternatives: {[v.get('type') for v in result.get('alternative_visualizations', [])]}")

def main():
    """Run all tests."""
    # Create test directory
    os.makedirs("test_visualizations", exist_ok=True)
    
    # Run tests
    test_2d_function_plot()
    test_3d_function_plot()
    test_derivative_plot()
    test_visualization_selector()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for generating visualizations directly from data without using LLM.
This works around any LLM-related issues.
"""

import os
import argparse
import numpy as np
from visualization.agent.viz_agent import VisualizationAgent
import uuid
from datetime import datetime

def test_scatter_plot():
    """Test generating a scatter plot directly."""
    print("Generating scatter plot...")
    
    # Ensure directory exists
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize the agent
    viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": False})
    
    # Data from the example
    x_data = list(range(1, 11))
    y_data = [2, 3, 5, 4, 6, 7, 8, 7, 9, 10]
    
    # Create message
    message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "test_script",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "scatter",
            "parameters": {
                "x_data": x_data,
                "y_data": y_data,
                "title": "Scatter Plot - Direct Test",
                "x_label": "X Values",
                "y_label": "Y Values",
                "show_regression": True,
                "filename": "direct_scatter_test.png"
            }
        }
    }
    
    # Process the request
    result = viz_agent.process_message(message)
    
    # Print result
    if result.get("success", False):
        print("Scatter plot created successfully!")
        print(f"File path: {result.get('file_path')}")
        file_size = os.path.getsize(result.get('file_path'))
        print(f"File size: {file_size} bytes")
    else:
        print(f"Error creating scatter plot: {result.get('error', 'Unknown error')}")

def test_function_plot():
    """Test generating a 2D function plot directly."""
    print("Generating function plot...")
    
    # Ensure directory exists
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize the agent
    viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": False})
    
    # Create message
    message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "test_script",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "function_2d",
            "parameters": {
                "expression": "sin(x) + cos(x)",
                "x_range": [-6.28, 6.28],  # -2π to 2π
                "title": "Sine plus Cosine Function - Direct Test",
                "x_label": "x",
                "y_label": "f(x) = sin(x) + cos(x)",
                "filename": "direct_function_test.png"
            }
        }
    }
    
    # Process the request
    result = viz_agent.process_message(message)
    
    # Print result
    if result.get("success", False):
        print("Function plot created successfully!")
        print(f"File path: {result.get('file_path')}")
        file_size = os.path.getsize(result.get('file_path'))
        print(f"File size: {file_size} bytes")
    else:
        print(f"Error creating function plot: {result.get('error', 'Unknown error')}")

def test_histogram():
    """Test generating a histogram directly."""
    print("Generating histogram...")
    
    # Ensure directory exists
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize the agent
    viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": False})
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    data = np.random.normal(0, 1, 1000).tolist()
    
    # Create message
    message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "test_script",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "histogram",
            "parameters": {
                "data": data,
                "bins": 30,
                "title": "Normal Distribution Histogram - Direct Test",
                "x_label": "Value",
                "y_label": "Frequency",
                "filename": "direct_histogram_test.png"
            }
        }
    }
    
    # Process the request
    result = viz_agent.process_message(message)
    
    # Print result
    if result.get("success", False):
        print("Histogram created successfully!")
        print(f"File path: {result.get('file_path')}")
        file_size = os.path.getsize(result.get('file_path'))
        print(f"File size: {file_size} bytes")
    else:
        print(f"Error creating histogram: {result.get('error', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="Test visualization generation directly")
    parser.add_argument("--type", choices=["scatter", "function", "histogram", "all"], 
                      default="all", help="Type of visualization to test")
    args = parser.parse_args()
    
    if args.type == "scatter" or args.type == "all":
        test_scatter_plot()
        print()
        
    if args.type == "function" or args.type == "all":
        test_function_plot()
        print()
        
    if args.type == "histogram" or args.type == "all":
        test_histogram()
        print()
        
    if args.type == "all":
        print("All tests completed.")

if __name__ == "__main__":
    main() 
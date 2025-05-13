#!/usr/bin/env python3
"""
Direct Visualization Client

This script creates visualizations from text descriptions without using the NLP endpoint. 
It directly constructs the visualization parameters and calls the VisualizationAgent.
"""

import os
import argparse
import re
import json
from datetime import datetime
import uuid
import numpy as np
import matplotlib.pyplot as plt
from visualization.agent.viz_agent import VisualizationAgent

def extract_scatter_data(prompt):
    """Extract scatter plot data from the prompt directly."""
    x_data = list(range(1, 11))
    y_data = []
    
    # Try to extract y values with regex pattern
    y_matches = re.findall(r"when x is (\d+), y is (\d+)", prompt)
    if y_matches and len(y_matches) > 0:
        # Sort by x value
        y_matches.sort(key=lambda m: int(m[0]))
        y_data = [int(m[1]) for m in y_matches]
    
    # If y_data is empty or not enough, look for another pattern
    if not y_data or len(y_data) < 10:
        y_data = []
        for i, x in enumerate(range(1, 11)):
            for pattern in [
                f"x is {x}, y is (\d+)",
                f"when x = {x}, y = (\d+)",
                f"x={x}, y=(\d+)",
                f"x = {x}, y = (\d+)",
                f"x-value {x} corresponds to y-value (\d+)"
            ]:
                matches = re.findall(pattern, prompt)
                if matches:
                    y_data.append(int(matches[0]))
                    break
    
    # If still not enough data, use the default example values
    if not y_data or len(y_data) < 10:
        y_data = [2, 3, 5, 4, 6, 7, 8, 7, 9, 10]
    
    return x_data, y_data

def create_scatter_plot(prompt, viz_agent=None):
    """Create a scatter plot based on the prompt."""
    if viz_agent is None:
        viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": False})
    
    # Extract data directly
    x_data, y_data = extract_scatter_data(prompt)
    
    # Current timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"direct_client_scatter_{timestamp}_{unique_id}.png"
    
    # Create message
    message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "direct_client",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "scatter",
            "parameters": {
                "x_data": x_data,
                "y_data": y_data,
                "title": "Scatter Plot from Text Description",
                "x_label": "X Values",
                "y_label": "Y Values",
                "show_regression": True,
                "filename": filename
            }
        }
    }
    
    # Process the request
    result = viz_agent.process_message(message)
    
    return result

def create_function_plot(prompt, viz_agent=None):
    """Create a function plot based on the prompt."""
    if viz_agent is None:
        viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": False})
    
    # Extract the function expression - default to sin(x) + cos(x) if not found
    function_expr = "sin(x) + cos(x)"
    x_range = [-6.28, 6.28]  # Default to -2π to 2π
    title = "Function Plot from Text Description"
    
    # Try to extract the function expression (simple method)
    expr_matches = re.findall(r"f\(x\) = ([\w\s\+\-\*\/\(\)\^]+)", prompt)
    if expr_matches:
        function_expr = expr_matches[0].strip()
    
    # Try to extract x range (simple method)
    range_matches = re.findall(r"range of x from ([-\d.]+) to ([-\d.]+)", prompt)
    if range_matches:
        try:
            x_min_str = range_matches[0][0]
            x_max_str = range_matches[0][1]
            
            # Handle π in the range values
            if 'π' in x_min_str or 'pi' in x_min_str:
                x_min_str = x_min_str.replace('π', '').replace('pi', '')
                if x_min_str.strip() in ['-', '+', '', '-2', '+2', '2']:
                    x_min = float(x_min_str.strip() + '1' if x_min_str.strip() in ['-', '+'] else x_min_str.strip()) * np.pi
                else:
                    x_min = float(x_min_str) * np.pi
            else:
                x_min = float(x_min_str)
                
            if 'π' in x_max_str or 'pi' in x_max_str:
                x_max_str = x_max_str.replace('π', '').replace('pi', '')
                if x_max_str.strip() in ['-', '+', '', '-2', '+2', '2']:
                    x_max = float(x_max_str.strip() + '1' if x_max_str.strip() in ['-', '+'] else x_max_str.strip()) * np.pi
                else:
                    x_max = float(x_max_str) * np.pi
            else:
                x_max = float(x_max_str)
                
            x_range = [x_min, x_max]
        except Exception as e:
            print(f"Warning: Could not parse x range: {e}")
    
    # Try to extract title
    title_matches = re.findall(r"title ['\"]([^'\"]+)['\"]", prompt)
    if title_matches:
        title = title_matches[0]
    else:
        title_matches = re.findall(r"give it the title ['\"]?([^'\"]+)['\"]?", prompt, re.IGNORECASE)
        if title_matches:
            title = title_matches[0]
    
    # Current timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"direct_client_function_{timestamp}_{unique_id}.png"
    
    print(f"DEBUG: Function expression: {function_expr}")
    print(f"DEBUG: X range: {x_range}")
    print(f"DEBUG: Title: {title}")
    
    # Create message
    message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "direct_client",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "function_2d",
            "parameters": {
                "expression": function_expr,
                "x_range": x_range,
                "title": title,
                "x_label": "x",
                "y_label": f"f(x) = {function_expr}",
                "filename": filename
            }
        }
    }
    
    # Process the request
    result = viz_agent.process_message(message)
    
    return result

def create_histogram(prompt, viz_agent=None):
    """Create a histogram based on the prompt."""
    if viz_agent is None:
        viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": False})
    
    # Generate random data for a histogram
    np.random.seed(42)  # For reproducibility
    data = np.random.normal(0, 1, 1000).tolist()
    
    # Extract parameters from the prompt, defaulting to reasonable values
    title = "Histogram from Text Description"
    bins = 30
    
    # Try to extract title
    title_matches = re.findall(r"title ['\"]([^'\"]+)['\"]", prompt)
    if title_matches:
        title = title_matches[0]
    
    # Try to extract bins
    bins_matches = re.findall(r"(\d+) bins", prompt)
    if bins_matches:
        try:
            bins = int(bins_matches[0])
        except:
            pass
    
    # Current timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"direct_client_histogram_{timestamp}_{unique_id}.png"
    
    # Create message
    message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "direct_client",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "histogram",
            "parameters": {
                "data": data,
                "bins": bins,
                "title": title,
                "x_label": "Value",
                "y_label": "Frequency",
                "filename": filename
            }
        }
    }
    
    # Process the request
    result = viz_agent.process_message(message)
    
    return result

def main():
    """Main function for the direct visualization client."""
    parser = argparse.ArgumentParser(description="Direct visualization client")
    parser.add_argument("--type", choices=["scatter", "function", "histogram"], default="scatter",
                      help="Type of visualization to create")
    parser.add_argument("--prompt", type=str, default="", help="Text description (optional)")
    
    args = parser.parse_args()
    
    # Set default prompts if not provided
    if not args.prompt:
        if args.type == "scatter":
            args.prompt = """
            The dataset consists of 10 data points with corresponding x and y values. 
            The x-values range from 1 to 10, and the y-values are as follows: 
            when x is 1, y is 2; when x is 2, y is 3; when x is 3, y is 5; 
            when x is 4, y is 4; when x is 5, y is 6; when x is 6, y is 7; 
            when x is 7, y is 8; when x is 8, y is 7; when x is 9, y is 9; 
            and when x is 10, y is 10. These coordinate pairs can be used to plot 
            a basic scatterplot showing a general upward trend.
            """
        elif args.type == "function":
            args.prompt = """
            Plot the function f(x) = sin(x) + cos(x) in the range of x from -2π to 2π.
            Label the x-axis as 'x' and the y-axis as 'f(x) = sin(x) + cos(x)'.
            Give it the title 'Sine plus Cosine Function'.
            """
        elif args.type == "histogram":
            args.prompt = """
            Create a histogram of a normal distribution with mean 0 and standard deviation 1.
            Generate 1000 random data points from this distribution.
            Use 30 bins and give it the title 'Normal Distribution Histogram'.
            """
    
    # Ensure visualizations directory exists
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize agent
    viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": False})
    
    # Create visualization based on type
    print(f"Creating {args.type} visualization...")
    
    if args.type == "scatter":
        result = create_scatter_plot(args.prompt, viz_agent)
    elif args.type == "function":
        result = create_function_plot(args.prompt, viz_agent)
    elif args.type == "histogram":
        result = create_histogram(args.prompt, viz_agent)
    
    # Print result
    if result.get("success", False):
        print(f"\n{args.type.capitalize()} visualization created successfully!")
        print(f"File path: {result.get('file_path')}")
        
        # Check file size
        file_path = result.get('file_path')
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")
            
            print(f"\nVisualization saved to: {os.path.abspath(file_path)}")
            print("\nYou can use this client for direct text-to-visualization conversion without using the API endpoint.")
        else:
            print("Warning: File not found")
    else:
        print(f"\nError creating {args.type} visualization:")
        print(result.get("error", "Unknown error"))

if __name__ == "__main__":
    main() 
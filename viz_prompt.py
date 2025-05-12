#!/usr/bin/env python3

"""
Simple visualization script that accepts a mathematical expression as input.
Example usage: python viz_prompt.py "sin(x) * cos(x)"
"""

import os
import sys
import json
from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent

def main():
    # Check if expression was provided
    if len(sys.argv) < 2:
        print("Usage: python viz_prompt.py \"<mathematical_expression>\"")
        print("Example: python viz_prompt.py \"sin(x) * cos(x)\"")
        return
    
    # Get the expression from command line argument
    expression = sys.argv[1]
    
    # Initialize the advanced visualization agent
    agent = AdvancedVisualizationAgent({
        "storage_dir": "output_visualizations",
        "use_database": False,
        "default_format": "png"
    })
    
    # Create output directory
    os.makedirs("output_visualizations", exist_ok=True)
    
    # Create visualization request
    message = {
        "header": {
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "function_2d",
            "parameters": {
                "expression": expression,
                "x_range": (-10, 10),
                "title": f"Plot of {expression}",
                "save": True
            }
        }
    }
    
    # Process message
    print(f"Visualizing: {expression}")
    result = agent.process_message(message)
    
    if result.get('success'):
        print(f"Visualization saved to: {result.get('file_path')}")
        
        # Open the file using the default application
        if 'file_path' in result and os.path.exists(result['file_path']):
            if sys.platform == 'darwin':  # macOS
                os.system(f"open {result['file_path']}")
            elif sys.platform == 'win32':  # Windows
                os.system(f"start {result['file_path']}")
            else:  # Linux or other
                os.system(f"xdg-open {result['file_path']}")
    else:
        print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    main() 
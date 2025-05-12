#!/usr/bin/env python3

"""
Simple script to test the visualization agent with a mathematical prompt.
"""

import os
import sys
import json
import matplotlib.pyplot as plt

# Add parent directory to Python path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from visualization.agent.viz_agent import VisualizationAgent
from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent

def main():
    # Initialize the advanced visualization agent (provides more capabilities)
    agent = AdvancedVisualizationAgent({
        "storage_dir": "output_visualizations",
        "use_database": False,
        "default_format": "png"
    })
    
    # Create a temporary output directory
    os.makedirs("output_visualizations", exist_ok=True)
    
    # The prompt: "plot the sin(x) cos(x)"
    # Process this by creating a message for the agent
    message = {
        "header": {
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "function_2d",
            "parameters": {
                "expression": "sin(x) * cos(x)",
                "x_range": (-10, 10),
                "title": "Plot of sin(x)cos(x)",
                "save": True
            }
        }
    }
    
    # Process message
    print("Processing visualization request...")
    result = agent.process_message(message)
    
    # Display result
    print(f"Result: {'Success' if result.get('success') else 'Failed'}")
    
    if result.get('success'):
        print(f"Visualization saved to: {result.get('file_path')}")
        
        # If the image has a base64 representation, we could display it
        if 'base64_image' in result:
            print("Base64 image available in the result")
        
        # Open the file using the default application
        if 'file_path' in result and os.path.exists(result['file_path']):
            print(f"Opening visualization file: {result['file_path']}")
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
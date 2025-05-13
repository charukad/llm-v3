#!/usr/bin/env python3
"""
Test script for the VisualizationAgent
"""

import os
import uuid
from datetime import datetime
import numpy as np
from visualization.agent.viz_agent import VisualizationAgent

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

# Create agent config
config = {
    "storage_dir": "visualizations",
    "use_database": False
}

# Initialize the agent
viz_agent = VisualizationAgent(config)

def test_agent():
    """Test the visualization agent with different visualization types"""
    
    # Get agent capabilities
    capabilities = viz_agent.get_capabilities()
    print("Agent Capabilities:")
    print(f"  Supported Types: {capabilities['supported_types']}")
    print(f"  Supported Formats: {capabilities['supported_formats']}")
    print()
    
    # Test 2D function visualization
    print("Testing 2D Function Visualization...")
    function_2d_message = {
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
                "expression": "sin(x)",
                "x_range": (-5, 5),
                "title": "Sine Function Test",
                "x_label": "x",
                "y_label": "sin(x)",
                "filename": "agent_test_sine.png"
            }
        }
    }
    
    result = viz_agent.process_message(function_2d_message)
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print()
    
    # Test multiple functions visualization
    print("Testing Multiple 2D Functions Visualization...")
    functions_2d_message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "test_script",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "functions_2d",
            "parameters": {
                "expressions": ["sin(x)", "cos(x)", "x^2/5"],
                "labels": ["sin(x)", "cos(x)", "x²/5"],
                "x_range": (-5, 5),
                "title": "Multiple Functions Test",
                "filename": "agent_test_multiple.png"
            }
        }
    }
    
    result = viz_agent.process_message(functions_2d_message)
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print()
    
    # Test 3D function visualization
    print("Testing 3D Function Visualization...")
    function_3d_message = {
        "header": {
            "message_id": str(uuid.uuid4()),
            "sender": "test_script",
            "recipient": "visualization_agent",
            "timestamp": datetime.now().isoformat(),
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": "function_3d",
            "parameters": {
                "expression": "sin(sqrt(x^2 + y^2))",
                "x_range": (-5, 5),
                "y_range": (-5, 5),
                "title": "3D Sinc Function Test",
                "filename": "agent_test_3d.png"
            }
        }
    }
    
    result = viz_agent.process_message(function_3d_message)
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print()
    
    # Test histogram visualization
    print("Testing Histogram Visualization...")
    data = np.random.normal(0, 1, 1000).tolist()  # Convert to list for JSON serialization
    histogram_message = {
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
                "title": "Normal Distribution Test",
                "filename": "agent_test_histogram.png"
            }
        }
    }
    
    result = viz_agent.process_message(histogram_message)
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print()

if __name__ == "__main__":
    print("Testing VisualizationAgent...")
    test_agent()
    print("Test completed. Check the 'visualizations' directory for output files.") 
#!/usr/bin/env python3
"""
Debug script for multiple functions visualization
"""

import os
from visualization.plotting.plot_2d import plot_multiple_functions_2d

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

# Test with direct function call
print("Testing plot_multiple_functions_2d directly...")
result = plot_multiple_functions_2d(
    functions=["sin(x)", "cos(x)", "x**2/5"],  # Using ** for power instead of ^
    labels=["sin(x)", "cos(x)", "x²/5"],
    x_range=(-5, 5),
    title="Multiple Functions Direct",
    save_path="visualizations/debug_multiple_direct.png"
)

print(f"Direct call result: {result['success']}")
if result['success']:
    print(f"File Path: {result['file_path']}")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")

# Import agent to test with message
from visualization.agent.viz_agent import VisualizationAgent
import uuid
from datetime import datetime

# Create agent config
config = {
    "storage_dir": "visualizations",
    "use_database": False
}

# Initialize the agent
viz_agent = VisualizationAgent(config)

# Test with agent message
print("\nTesting with agent message...")
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
            "expressions": ["sin(x)", "cos(x)", "x**2/5"],  # Using ** for power instead of ^
            "labels": ["sin(x)", "cos(x)", "x²/5"],
            "x_range": (-5, 5),
            "title": "Multiple Functions Test",
            "filename": "agent_test_multiple.png"
        }
    }
}

# Print the message for debugging
print("Message body:")
import json
print(json.dumps(functions_2d_message["body"], indent=2, default=str))

# Debug wrapper
def debug_process_message(agent, message):
    print("\nDebugging process_message...")
    body = message.get("body", {})
    viz_type = body.get("visualization_type")
    params = body.get("parameters", {})
    print(f"Visualization type: {viz_type}")
    print(f"Parameters: {params}")
    
    # Check if visualization type is supported
    if viz_type not in agent.supported_types:
        print(f"Error: Unsupported visualization type: {viz_type}")
        return None
        
    # Get the visualization function
    viz_func = agent.supported_types[viz_type]
    print(f"Using function: {viz_func.__name__}")
    
    # Print expressions parameter
    expressions = params.get("expressions")
    print(f"Expressions: {expressions}")
    print(f"Type of expressions: {type(expressions)}")
    if expressions:
        print(f"Length of expressions: {len(expressions)}")
    
    # Call the function directly
    print("Calling function...")
    try:
        result = viz_func(params)
        print(f"Result: {result}")
        return result
    except Exception as e:
        import traceback
        print(f"Exception: {e}")
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}

# Call with debug wrapper
result = debug_process_message(viz_agent, functions_2d_message)
# result = viz_agent.process_message(functions_2d_message)

print(f"Agent message result: {result['success'] if result else 'None'}")
if result and result['success']:
    print(f"File Path: {result['file_path']}")
else:
    print(f"Error: {result.get('error', 'Unknown error') if result else 'None'}") 
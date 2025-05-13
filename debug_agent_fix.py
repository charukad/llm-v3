#!/usr/bin/env python3
"""
Debug script with a patched version of the visualization agent
"""

import os
import uuid
import copy
from datetime import datetime
import numpy as np
import traceback
from visualization.agent.viz_agent import VisualizationAgent
from visualization.plotting.plot_2d import plot_multiple_functions_2d

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

# Create a subclass of VisualizationAgent with a patched version of _plot_multiple_2d_functions
class PatchedVisualizationAgent(VisualizationAgent):
    def _plot_multiple_2d_functions(self, parameters: dict) -> dict:
        """Patched version with detailed debugging for multiple 2D functions plotting."""
        print("\n=== Inside _plot_multiple_2d_functions ===")
        print(f"Parameters: {parameters}")
        
        try:
            # Extract parameters
            expressions = parameters.get("expressions")
            print(f"Expressions: {expressions}")
            print(f"Type of expressions: {type(expressions)}")
            
            if not expressions:
                print("Error: Missing expressions parameter")
                return {"success": False, "error": "Missing required parameter: expressions"}
            
            # Handle optional parameters
            labels = parameters.get("labels")
            print(f"Labels: {labels}")
            
            x_range = parameters.get("x_range", (-10, 10))
            print(f"X Range: {x_range}")
            
            num_points = parameters.get("num_points", 1000)
            title = parameters.get("title", "Multiple Functions")
            x_label = parameters.get("x_label", "x")
            y_label = parameters.get("y_label", "y")
            show_grid = parameters.get("show_grid", True)
            figsize = parameters.get("figsize", (8, 6))
            dpi = parameters.get("dpi", self.default_dpi)
            colors = parameters.get("colors")
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"functions_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            print(f"Save path: {save_path}")
            
            # Attempt direct call to plot_multiple_functions_2d
            print("\nDirect call to plot_multiple_functions_2d:")
            try:
                direct_result = plot_multiple_functions_2d(
                    functions=expressions,
                    labels=labels,
                    x_range=x_range,
                    num_points=num_points,
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    show_grid=show_grid,
                    figsize=figsize,
                    dpi=dpi,
                    colors=colors,
                    save_path=save_path
                )
                print(f"Direct call result: {direct_result}")
                return direct_result
            except Exception as direct_e:
                print(f"Direct call exception: {direct_e}")
                print(traceback.format_exc())
            
            # If we're here, the direct call failed, so try with modified parameters
            print("\nTrying with modified parameters...")
            
            # Make a copy of the parameters to modify
            modified_params = copy.deepcopy(parameters)
            
            # Try with different parameter combinations
            modified_params["save_path"] = save_path  # Try explicit save_path parameter
            
            if isinstance(x_range, list):
                modified_params["x_range"] = tuple(x_range)  # Convert list to tuple
                
            # Fix power symbol in expressions if needed
            modified_expressions = []
            for expr in expressions:
                if "^" in expr:
                    # Replace ^ with ** for Python power
                    modified_expr = expr.replace("^", "**")
                    modified_expressions.append(modified_expr)
                else:
                    modified_expressions.append(expr)
            
            modified_params["functions"] = modified_expressions
            
            print(f"Modified parameters: {modified_params}")
            
            try:
                modified_result = plot_multiple_functions_2d(
                    functions=modified_expressions,
                    labels=labels,
                    x_range=tuple(x_range) if isinstance(x_range, list) else x_range,
                    num_points=num_points,
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    show_grid=show_grid,
                    figsize=figsize,
                    dpi=dpi,
                    colors=colors,
                    save_path=save_path
                )
                print(f"Modified call result: {modified_result}")
                return modified_result
            except Exception as modified_e:
                print(f"Modified call exception: {modified_e}")
                print(traceback.format_exc())
                
            # If all attempts fail, return error
            return {"success": False, "error": f"Error in multiple 2D function plotting: Failed to plot with multiple attempts"}
            
        except Exception as e:
            print(f"Exception in _plot_multiple_2d_functions: {e}")
            print(traceback.format_exc())
            return {"success": False, "error": f"Error in multiple 2D function plotting: {str(e)}"}

# Create agent config
config = {
    "storage_dir": "visualizations",
    "use_database": False
}

# Initialize the patched agent
viz_agent = PatchedVisualizationAgent(config)

# Test multiple functions visualization
print("Testing Multiple 2D Functions Visualization with patched agent...")
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
            "expressions": ["sin(x)", "cos(x)", "x**2/5"],
            "labels": ["sin(x)", "cos(x)", "x²/5"],
            "x_range": (-5, 5),
            "title": "Multiple Functions Test",
            "filename": "patched_agent_test_multiple.png"
        }
    }
}

result = viz_agent.process_message(functions_2d_message)
print(f"\nFinal result: {result}")

if result["success"]:
    print(f"File Path: {result['file_path']}")
else:
    print(f"Error: {result.get('error', 'Unknown error')}") 
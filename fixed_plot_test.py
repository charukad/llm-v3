#!/usr/bin/env python3
"""
Test script with a fixed version of the plot_multiple_functions_2d function
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from typing import Dict, Any, Tuple, Optional, List, Union
import io
import base64
import uuid
from datetime import datetime

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

def fixed_plot_multiple_functions_2d(
    functions: List[Union[sp.Expr, str]],
    labels: Optional[List[str]] = None,
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000,
    title: str = "Multiple Functions",
    x_label: str = "x",
    y_label: str = "y",
    show_grid: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 100,
    colors: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fixed version of plot_multiple_functions_2d that handles None colors correctly.
    """
    print(f"Fixed function called with:")
    print(f"  functions: {functions}")
    print(f"  labels: {labels}")
    print(f"  colors: {colors}")
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    
    # Setup default colors if not provided
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Make sure colors is a list with enough elements
    colors = colors * (len(functions) // len(colors) + 1)
    
    # Plot each function
    plot_data = []
    for i, func_expr in enumerate(functions):
        try:
            # Convert string expression to SymPy if needed
            if isinstance(func_expr, str):
                x = sp.symbols('x')
                # Replace ** with ^ for sympy
                if "**" in func_expr:
                    func_expr = func_expr.replace("**", "^")
                func_expr = sp.sympify(func_expr)
                
            # Convert SymPy expression to NumPy function
            x = sp.symbols('x')
            f = sp.lambdify(x, func_expr, "numpy")
            
            # Compute y values
            y_vals = f(x_vals)
            
            # Check for infinities or NaN values
            mask = np.isfinite(y_vals)
            
            # Plot the function
            label = labels[i] if labels and i < len(labels) else f"f_{i+1}(x)"
            ax.plot(x_vals[mask], y_vals[mask], color=colors[i], label=label)
            
            # Store data for return
            plot_data.append({
                "expression": str(func_expr),
                "expression_latex": sp.latex(func_expr),
                "label": label,
                "x_range": x_range,
                "valid_points": np.sum(mask)
            })
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to plot function {func_expr}: {str(e)}"
            }
    
    # Add grid and labels
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Add title
    ax.set_title(title)
    
    # Add legend if we have multiple functions
    if len(functions) > 1:
        ax.legend()
    
    # Save or encode the figure
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path)
        plt.close(fig)
        
        return {
            "success": True,
            "plot_type": "2d_multiple_functions",
            "file_path": save_path,
            "data": plot_data
        }
    else:
        # Convert to base64 for embedding in web applications
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        
        return {
            "success": True,
            "plot_type": "2d_multiple_functions",
            "base64_image": image_base64,
            "data": plot_data
        }

# Test the fixed function
def test_fixed_function():
    print("Testing fixed_plot_multiple_functions_2d...")
    
    result = fixed_plot_multiple_functions_2d(
        functions=["sin(x)", "cos(x)", "x**2/5"],
        labels=["sin(x)", "cos(x)", "x²/5"],
        x_range=(-5, 5),
        title="Multiple Functions Fixed Test",
        save_path="visualizations/fixed_multiple_functions.png"
    )
    
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_fixed_function() 
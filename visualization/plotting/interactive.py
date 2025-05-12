"""
Interactive visualization functions using Plotly.

This module provides functions for creating interactive visualizations 
using Plotly, which allows users to zoom, pan, and explore data.
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import json
import uuid
from datetime import datetime

def interactive_function_2d(
    function_expr: Union[sp.Expr, str],
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000,
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    show_grid: bool = True,
    width: int = 800,
    height: int = 500,
    template: str = "plotly_white",
    line_color: str = "#1f77b4",
    save_path: Optional[str] = None,
    include_html: bool = True
) -> Dict[str, Any]:
    """
    Create an interactive 2D function plot using Plotly.
    
    Args:
        function_expr: SymPy expression or string to plot
        x_range: Tuple with (min_x, max_x) values
        num_points: Number of points to sample for plotting
        title: Plot title (defaults to LaTeX representation if None)
        x_label: Label for x-axis
        y_label: Label for y-axis
        show_grid: Whether to display grid
        width: Width of the plot in pixels
        height: Height of the plot in pixels
        template: Plotly template to use
        line_color: Color of the function line
        save_path: Path to save the figure (if None, won't be saved)
        include_html: Whether to include HTML in the result
        
    Returns:
        Dictionary with plot information
    """
    try:
        # Convert string expression to SymPy if needed
        if isinstance(function_expr, str):
            x = sp.symbols('x')
            expr = sp.sympify(function_expr)
        else:
            expr = function_expr
            x = sp.symbols('x')
        
        # Generate x values
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        
        # Convert SymPy expression to NumPy function
        f = sp.lambdify(x, expr, "numpy")
        
        # Calculate y values
        y_vals = f(x_vals)
        
        # Check for infinities or NaN values
        mask = np.isfinite(y_vals)
        x_vals_filtered = x_vals[mask]
        y_vals_filtered = y_vals[mask]
        
        # Create figure
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=x_vals_filtered, 
                y=y_vals_filtered,
                mode='lines',
                line=dict(color=line_color, width=2),
                name=f"f(x) = {sp.latex(expr)}"
            )
        )
        
        # Set title
        if title:
            fig.update_layout(title=title)
        else:
            fig.update_layout(title=f"f(x) = {sp.latex(expr)}")
        
        # Set axes labels
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        # Set grid
        fig.update_layout(
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid)
        )
        
        # Set figure size and template
        fig.update_layout(
            width=width,
            height=height,
            template=template
        )
        
        # Add hover mode
        fig.update_layout(hovermode="closest")
        
        # Convert to HTML
        html_content = None
        if include_html:
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save as both HTML and JSON
            html_path = save_path if save_path.endswith('.html') else f"{save_path}.html"
            json_path = os.path.splitext(html_path)[0] + '.json'
            
            # Save HTML
            with open(html_path, 'w') as f:
                f.write(fig.to_html(include_plotlyjs='cdn'))
            
            # Save JSON
            with open(json_path, 'w') as f:
                f.write(fig.to_json())
        
        # Return result
        result = {
            "success": True,
            "plot_type": "interactive_2d_function",
            "data": {
                "expression": str(expr),
                "expression_latex": sp.latex(expr),
                "x_range": x_range,
                "valid_points": int(np.sum(mask))
            }
        }
        
        # Add html if include_html is True
        if include_html:
            result["html"] = html_content
        
        # Add paths if saved
        if save_path:
            result["html_path"] = html_path
            result["json_path"] = json_path
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create interactive 2D function plot: {str(e)}"
        }

def interactive_function_3d(
    function_expr: Union[sp.Expr, str],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    num_points: int = 50,
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "z",
    width: int = 800,
    height: int = 600,
    template: str = "plotly_white",
    colorscale: str = "viridis",
    save_path: Optional[str] = None,
    include_html: bool = True
) -> Dict[str, Any]:
    """
    Create an interactive 3D function plot using Plotly.
    
    Args:
        function_expr: SymPy expression or string to plot (must be a function of x and y)
        x_range: Tuple with (min_x, max_x) values
        y_range: Tuple with (min_y, max_y) values
        num_points: Number of points to sample in each dimension
        title: Plot title (defaults to LaTeX representation if None)
        x_label: Label for x-axis
        y_label: Label for y-axis
        z_label: Label for z-axis
        width: Width of the plot in pixels
        height: Height of the plot in pixels
        template: Plotly template to use
        colorscale: Colorscale for the surface
        save_path: Path to save the figure (if None, won't be saved)
        include_html: Whether to include HTML in the result
        
    Returns:
        Dictionary with plot information
    """
    try:
        # Convert string expression to SymPy if needed
        if isinstance(function_expr, str):
            x, y = sp.symbols('x y')
            expr = sp.sympify(function_expr)
        else:
            expr = function_expr
            x, y = sp.symbols('x y')
        
        # Generate x and y values
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Convert SymPy expression to NumPy function
        f = sp.lambdify((x, y), expr, "numpy")
        
        # Calculate z values
        Z = f(X, Y)
        
        # Check for infinities or NaN values
        mask = np.isfinite(Z)
        if not np.any(mask):
            return {
                "success": False,
                "error": "No finite values in the specified range"
            }
        
        # Replace infinities with NaN
        Z = np.where(mask, Z, np.nan)
        
        # Create figure
        fig = go.Figure(data=[
            go.Surface(
                x=X, 
                y=Y, 
                z=Z,
                colorscale=colorscale,
                showscale=True
            )
        ])
        
        # Set title
        if title:
            fig.update_layout(title=title)
        else:
            fig.update_layout(title=f"f(x,y) = {sp.latex(expr)}")
        
        # Set axes labels
        fig.update_layout(scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ))
        
        # Set figure size and template
        fig.update_layout(
            width=width,
            height=height,
            template=template
        )
        
        # Add hover mode
        fig.update_layout(hovermode="closest")
        
        # Convert to HTML
        html_content = None
        if include_html:
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save as both HTML and JSON
            html_path = save_path if save_path.endswith('.html') else f"{save_path}.html"
            json_path = os.path.splitext(html_path)[0] + '.json'
            
            # Save HTML
            with open(html_path, 'w') as f:
                f.write(fig.to_html(include_plotlyjs='cdn'))
            
            # Save JSON
            with open(json_path, 'w') as f:
                f.write(fig.to_json())
        
        # Return result
        result = {
            "success": True,
            "plot_type": "interactive_3d_function",
            "data": {
                "expression": str(expr),
                "expression_latex": sp.latex(expr),
                "x_range": x_range,
                "y_range": y_range,
                "finite_points": int(np.sum(mask))
            }
        }
        
        # Add html if include_html is True
        if include_html:
            result["html"] = html_content
        
        # Add paths if saved
        if save_path:
            result["html_path"] = html_path
            result["json_path"] = json_path
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create interactive 3D function plot: {str(e)}"
        }

def interactive_scatter_3d(
    x_data: List[float],
    y_data: List[float],
    z_data: List[float],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
    width: int = 800,
    height: int = 600,
    template: str = "plotly_white",
    colorscale: str = "viridis",
    marker_size: int = 5,
    save_path: Optional[str] = None,
    include_html: bool = True
) -> Dict[str, Any]:
    """
    Create an interactive 3D scatter plot using Plotly.
    
    Args:
        x_data: List of x coordinates
        y_data: List of y coordinates
        z_data: List of z coordinates
        labels: Optional list of point labels
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        z_label: Label for z-axis
        width: Width of the plot in pixels
        height: Height of the plot in pixels
        template: Plotly template to use
        colorscale: Colorscale for the points
        marker_size: Size of the markers
        save_path: Path to save the figure (if None, won't be saved)
        include_html: Whether to include HTML in the result
        
    Returns:
        Dictionary with plot information
    """
    try:
        # Check for valid data
        if len(x_data) != len(y_data) or len(x_data) != len(z_data):
            return {
                "success": False,
                "error": "x_data, y_data, and z_data must have the same length"
            }
        
        # Create figure
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=z_data,
                    colorscale=colorscale,
                    opacity=0.8
                ),
                text=labels
            )
        ])
        
        # Set title
        if title:
            fig.update_layout(title=title)
        
        # Set axes labels
        fig.update_layout(scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ))
        
        # Set figure size and template
        fig.update_layout(
            width=width,
            height=height,
            template=template
        )
        
        # Convert to HTML
        html_content = None
        if include_html:
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save as both HTML and JSON
            html_path = save_path if save_path.endswith('.html') else f"{save_path}.html"
            json_path = os.path.splitext(html_path)[0] + '.json'
            
            # Save HTML
            with open(html_path, 'w') as f:
                f.write(fig.to_html(include_plotlyjs='cdn'))
            
            # Save JSON
            with open(json_path, 'w') as f:
                f.write(fig.to_json())
        
        # Return result
        result = {
            "success": True,
            "plot_type": "interactive_scatter_3d",
            "data": {
                "num_points": len(x_data),
                "x_range": [min(x_data), max(x_data)],
                "y_range": [min(y_data), max(y_data)],
                "z_range": [min(z_data), max(z_data)]
            }
        }
        
        # Add html if include_html is True
        if include_html:
            result["html"] = html_content
        
        # Add paths if saved
        if save_path:
            result["html_path"] = html_path
            result["json_path"] = json_path
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create interactive 3D scatter plot: {str(e)}"
        }

def interactive_multivariate_function(
    function_expr: Union[sp.Expr, str],
    sliders: Dict[str, Dict[str, Any]],
    variable_x: str = "x",
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: str = "f(x)",
    width: int = 900,
    height: int = 600,
    template: str = "plotly_white",
    line_color: str = "#1f77b4",
    save_path: Optional[str] = None,
    include_html: bool = True
) -> Dict[str, Any]:
    """
    Create an interactive plot of a multivariate function with sliders for parameters.
    
    Args:
        function_expr: SymPy expression or string to plot
        sliders: Dictionary of sliders with variable names as keys and configuration as values
                 Each slider config should have 'min', 'max', 'step', and 'initial' keys
        variable_x: The variable to use for the x-axis
        x_range: Tuple with (min_x, max_x) values
        num_points: Number of points to sample for plotting
        title: Plot title
        x_label: Label for x-axis (defaults to variable_x if None)
        y_label: Label for y-axis
        width: Width of the plot in pixels
        height: Height of the plot in pixels
        template: Plotly template to use
        line_color: Color of the function line
        save_path: Path to save the figure (if None, won't be saved)
        include_html: Whether to include HTML in the result
        
    Returns:
        Dictionary with plot information
    """
    try:
        # Convert string expression to SymPy if needed
        if isinstance(function_expr, str):
            expr = sp.sympify(function_expr)
        else:
            expr = function_expr
        
        # Get all symbols in the expression
        symbols = list(expr.free_symbols)
        symbol_names = [str(symbol) for symbol in symbols]
        
        # Check that variable_x is in the symbols
        if variable_x not in symbol_names:
            return {
                "success": False,
                "error": f"Variable '{variable_x}' not found in the expression"
            }
        
        # Check that all slider variables are in the symbols
        for var_name in sliders.keys():
            if var_name not in symbol_names:
                return {
                    "success": False,
                    "error": f"Slider variable '{var_name}' not found in the expression"
                }
        
        # Get symbol for x variable
        x_symbol = [s for s in symbols if str(s) == variable_x][0]
        
        # Generate initial values dictionary for all variables
        initial_values = {}
        for var_name in symbol_names:
            if var_name == variable_x:
                continue  # Skip x variable
            elif var_name in sliders:
                initial_values[var_name] = sliders[var_name]['initial']
            else:
                # Use 0 as default for variables without sliders
                initial_values[var_name] = 0
        
        # Generate x values
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        
        # Create figures for each slider step
        fig = go.Figure()
        
        # Add initial trace
        def compute_trace(values):
            # Create a lambda function with all variables except x
            var_dict = {var_name: values[var_name] for var_name in values}
            
            # Substitute values into expression
            substituted_expr = expr.subs(var_dict)
            
            # Convert to numpy function of x only
            f = sp.lambdify(x_symbol, substituted_expr, "numpy")
            
            # Compute y values
            y_vals = f(x_vals)
            
            # Filter out infinities and NaNs
            mask = np.isfinite(y_vals)
            x_filtered = x_vals[mask]
            y_filtered = y_vals[mask]
            
            return x_filtered, y_filtered
        
        # Compute initial trace
        x_filtered, y_filtered = compute_trace(initial_values)
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=x_filtered, 
                y=y_filtered,
                mode='lines',
                line=dict(color=line_color, width=2),
                name=f"f({variable_x})"
            )
        )
        
        # Add sliders
        sliders_list = []
        
        # Create a slider for each variable
        for var_name, slider_config in sliders.items():
            slider = dict(
                active=0,
                currentvalue={"prefix": f"{var_name} = "},
                pad={"t": 50},
                steps=[]
            )
            
            # Generate steps
            min_val = slider_config['min']
            max_val = slider_config['max']
            step_val = slider_config['step']
            num_steps = int((max_val - min_val) / step_val) + 1
            
            for i in range(num_steps):
                value = min_val + i * step_val
                
                # Update values dictionary
                values = dict(initial_values)
                values[var_name] = value
                
                # Compute trace
                x_filtered, y_filtered = compute_trace(values)
                
                # Add step
                slider["steps"].append(
                    dict(
                        method="update",
                        args=[
                            {"x": [x_filtered], "y": [y_filtered]},
                            {"title": f"f({variable_x}) with {var_name} = {value}"}
                        ],
                        label=str(value)
                    )
                )
            
            sliders_list.append(slider)
        
        # Set layout with sliders
        fig.update_layout(sliders=sliders_list)
        
        # Set title
        if title:
            fig.update_layout(title=title)
        else:
            # Create title with initial values
            param_str = ", ".join([f"{var}={val}" for var, val in initial_values.items()])
            fig.update_layout(title=f"f({variable_x}, {param_str}) = {sp.latex(expr)}")
        
        # Set axes labels
        if x_label is None:
            x_label = variable_x
            
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        # Set figure size and template
        fig.update_layout(
            width=width,
            height=height,
            template=template
        )
        
        # Add hover mode
        fig.update_layout(hovermode="closest")
        
        # Convert to HTML
        html_content = None
        if include_html:
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save as both HTML and JSON
            html_path = save_path if save_path.endswith('.html') else f"{save_path}.html"
            json_path = os.path.splitext(html_path)[0] + '.json'
            
            # Save HTML
            with open(html_path, 'w') as f:
                f.write(fig.to_html(include_plotlyjs='cdn'))
            
            # Save JSON
            with open(json_path, 'w') as f:
                f.write(fig.to_json())
        
        # Return result
        result = {
            "success": True,
            "plot_type": "interactive_multivariate_function",
            "data": {
                "expression": str(expr),
                "expression_latex": sp.latex(expr),
                "x_variable": variable_x,
                "x_range": x_range,
                "parameters": {var: sliders[var]['initial'] for var in sliders},
                "all_symbols": symbol_names
            }
        }
        
        # Add html if include_html is True
        if include_html:
            result["html"] = html_content
        
        # Add paths if saved
        if save_path:
            result["html_path"] = html_path
            result["json_path"] = json_path
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create interactive multivariate function plot: {str(e)}"
        }

def generate_unique_filename(base_dir: str, prefix: str = "plot", extension: str = "html") -> str:
    """Generate a unique filename for saving plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(base_dir, f"{prefix}_{timestamp}_{unique_id}.{extension}")

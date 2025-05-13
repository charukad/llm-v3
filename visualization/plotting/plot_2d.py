import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from typing import Dict, Any, Tuple, Optional, List, Union
import io
import base64
import os
from datetime import datetime
import uuid

def plot_function_2d(
    function_expr: Union[sp.Expr, str], 
    x_range: Tuple[float, float] = (-10, 10), 
    num_points: int = 1000, 
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    show_grid: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 100,
    colors: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a 2D plot of a mathematical function.
    
    Args:
        function_expr: SymPy expression or string to plot
        x_range: Tuple with (min_x, max_x) values
        num_points: Number of points to sample for plotting
        title: Plot title (defaults to LaTeX representation if None)
        x_label: Label for x-axis
        y_label: Label for y-axis
        show_grid: Whether to display grid
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for the figure
        colors: List of colors for multiple functions
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with plot information, including base64 encoded image or path
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    
    # Convert string expression to SymPy if needed
    if isinstance(function_expr, str):
        try:
            x = sp.symbols('x')
            function_expr = sp.sympify(function_expr)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse expression: {str(e)}"
            }
    
    # Convert SymPy expression to NumPy function
    x = sp.symbols('x')
    
    # Handle multiple functions (if function_expr is a list)
    functions = [function_expr] if not isinstance(function_expr, list) else function_expr
    
    # Setup default colors if not provided
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        # Repeat colors if needed
        colors = colors * (len(functions) // len(colors) + 1)
    
    # Plot each function
    plot_data = []
    for i, func_expr in enumerate(functions):
        try:
            f = sp.lambdify(x, func_expr, "numpy")
            y_vals = f(x_vals)
            
            # Check for infinities or NaN values
            mask = np.isfinite(y_vals)
            
            # Plot the function
            ax.plot(x_vals[mask], y_vals[mask], color=colors[i % len(colors)])
            
            # Store data for return
            plot_data.append({
                "expression": str(func_expr),
                "expression_latex": sp.latex(func_expr),
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
    
    # Add title if provided, otherwise use LaTeX representation
    if title:
        ax.set_title(title)
    elif len(functions) == 1:
        ax.set_title(f"$f(x) = {sp.latex(functions[0])}$")
    else:
        ax.set_title("Multiple Functions")
    
    # Save or encode the figure
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path)
        plt.close(fig)
        
        return {
            "success": True,
            "plot_type": "2d_function",
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
            "plot_type": "2d_function",
            "base64_image": image_base64,
            "data": plot_data
        }

def plot_multiple_functions_2d(
    functions: List[Union[sp.Expr, str]],
    labels: Optional[List[str]] = None,
    x_range: Tuple[float, float] = (-10, 10),
    **kwargs
) -> Dict[str, Any]:
    """
    Plot multiple functions on the same graph.
    
    Args:
        functions: List of functions to plot
        labels: List of labels for the legend
        x_range: Tuple with (min_x, max_x) values
        **kwargs: Additional arguments passed to plot_function_2d
        
    Returns:
        Dictionary with plot information
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)), dpi=kwargs.get('dpi', 100))
    
    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], kwargs.get('num_points', 1000))
    
    # Setup colors
    colors = kwargs.get('colors', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    colors = colors * (len(functions) // len(colors) + 1)
    
    # Plot each function
    plot_data = []
    for i, func_expr in enumerate(functions):
        try:
            # Convert string expression to SymPy if needed
            if isinstance(func_expr, str):
                x = sp.symbols('x')
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
    if kwargs.get('show_grid', True):
        ax.grid(True, alpha=0.3)
    ax.set_xlabel(kwargs.get('x_label', 'x'))
    ax.set_ylabel(kwargs.get('y_label', 'y'))
    
    # Add title
    title = kwargs.get('title', 'Multiple Functions')
    ax.set_title(title)
    
    # Add legend if we have multiple functions
    if len(functions) > 1:
        ax.legend()
    
    # Save or encode the figure
    save_path = kwargs.get('save_path', None)
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

def generate_unique_filename(base_dir: str, prefix: str = "plot", extension: str = "png") -> str:
    """Generate a unique filename for saving plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(base_dir, f"{prefix}_{timestamp}_{unique_id}.{extension}")

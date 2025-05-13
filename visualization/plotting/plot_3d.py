import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp
from typing import Dict, Any, Tuple, Optional, List, Union
import io
import base64
import os
from datetime import datetime
import uuid

def plot_function_3d(
    function_expr: Union[sp.Expr, str], 
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    num_points: int = 50, 
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "z",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    view_angle: Tuple[float, float] = (30, 30),
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a 3D surface plot of a mathematical function f(x,y).
    
    Args:
        function_expr: SymPy expression or string to plot (must be a function of x and y)
        x_range: Tuple with (min_x, max_x) values
        y_range: Tuple with (min_y, max_y) values
        num_points: Number of points to sample in each dimension
        title: Plot title (defaults to LaTeX representation if None)
        x_label: Label for x-axis
        y_label: Label for y-axis
        z_label: Label for z-axis
        figsize: Figure size in inches (width, height)
        cmap: Colormap for the surface
        view_angle: Initial viewing angle (elevation, azimuth)
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with plot information, including base64 encoded image or path
    """
    # Setup figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate x and y values
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Convert string expression to SymPy if needed
    if isinstance(function_expr, str):
        try:
            x_sym, y_sym = sp.symbols('x y')
            function_expr = sp.sympify(function_expr)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse expression: {str(e)}"
            }
    
    # Convert SymPy expression to NumPy function
    x_sym, y_sym = sp.symbols('x y')
    
    try:
        f = sp.lambdify((x_sym, y_sym), function_expr, "numpy")
        Z = f(X, Y)
        
        # Check for infinities or NaN values
        mask = np.isfinite(Z)
        if not np.any(mask):
            return {
                "success": False,
                "error": "No finite values in the specified range"
            }
        
        # Replace infinities and NaNs with NaN for plotting
        Z = np.where(mask, Z, np.nan)
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, linewidth=0)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Add title if provided, otherwise use LaTeX representation
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"$f(x,y) = {sp.latex(function_expr)}$")
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "3d_function",
                "file_path": save_path,
                "data": {
                    "expression": str(function_expr),
                    "expression_latex": sp.latex(function_expr),
                    "x_range": x_range,
                    "y_range": y_range,
                    "finite_points": np.sum(mask)
                }
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
                "plot_type": "3d_function",
                "base64_image": image_base64,
                "data": {
                    "expression": str(function_expr),
                    "expression_latex": sp.latex(function_expr),
                    "x_range": x_range,
                    "y_range": y_range,
                    "finite_points": np.sum(mask)
                }
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to plot function: {str(e)}"
        }

def plot_parametric_3d(
    x_expr: Union[sp.Expr, str],
    y_expr: Union[sp.Expr, str],
    z_expr: Union[sp.Expr, str],
    t_range: Tuple[float, float] = (0, 2*np.pi),
    num_points: int = 1000,
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "z",
    figsize: Tuple[int, int] = (10, 8),
    color: str = 'blue',
    view_angle: Tuple[float, float] = (30, 30),
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a 3D parametric curve plot.
    
    Args:
        x_expr: SymPy expression or string for x(t)
        y_expr: SymPy expression or string for y(t)
        z_expr: SymPy expression or string for z(t)
        t_range: Tuple with (min_t, max_t) values
        num_points: Number of points to sample
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        z_label: Label for z-axis
        figsize: Figure size in inches (width, height)
        color: Color for the curve
        view_angle: Initial viewing angle (elevation, azimuth)
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with plot information, including base64 encoded image or path
    """
    # Setup figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate t values
    t_vals = np.linspace(t_range[0], t_range[1], num_points)
    
    # Process expressions
    exprs = []
    for expr in [x_expr, y_expr, z_expr]:
        if isinstance(expr, str):
            try:
                t = sp.symbols('t')
                expr = sp.sympify(expr)
                exprs.append(expr)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to parse expression: {str(e)}"
                }
        else:
            exprs.append(expr)
    
    # Convert SymPy expressions to NumPy functions
    t = sp.symbols('t')
    
    try:
        x_func = sp.lambdify(t, exprs[0], "numpy")
        y_func = sp.lambdify(t, exprs[1], "numpy")
        z_func = sp.lambdify(t, exprs[2], "numpy")
        
        # Compute values
        x_vals = x_func(t_vals)
        y_vals = y_func(t_vals)
        z_vals = z_func(t_vals)
        
        # Check for infinities or NaN values
        mask = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
        if not np.any(mask):
            return {
                "success": False,
                "error": "No finite values in the specified range"
            }
        
        # Plot the curve
        ax.plot(x_vals[mask], y_vals[mask], z_vals[mask], color=color)
        
        # Set labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Add title if provided
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Parametric Curve")
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "3d_parametric",
                "file_path": save_path,
                "data": {
                    "x_expression": str(exprs[0]),
                    "y_expression": str(exprs[1]),
                    "z_expression": str(exprs[2]),
                    "t_range": t_range,
                    "finite_points": np.sum(mask)
                }
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
                "plot_type": "3d_parametric",
                "base64_image": image_base64,
                "data": {
                    "x_expression": str(exprs[0]),
                    "y_expression": str(exprs[1]),
                    "z_expression": str(exprs[2]),
                    "t_range": t_range,
                    "finite_points": np.sum(mask)
                }
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to plot parametric curve: {str(e)}"
        }

def generate_unique_filename(base_dir: str, prefix: str = "plot", extension: str = "png") -> str:
    """Generate a unique filename for saving plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(base_dir, f"{prefix}_{timestamp}_{unique_id}.{extension}")

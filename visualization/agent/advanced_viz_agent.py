import os
import json
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import sympy as sp
import uuid
from datetime import datetime

# Local imports
from visualization.agent.viz_agent import VisualizationAgent
from math_processing.computation.sympy_wrapper import SymbolicProcessor

class AdvancedVisualizationAgent(VisualizationAgent):
    """
    Extended visualization agent with advanced capabilities for mathematical visualization.
    This agent integrates with the Mathematical Computation Agent for enhanced features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Advanced Visualization Agent.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        # Initialize the base visualization agent
        super().__init__(config)
        
        # Initialize symbolic processor for mathematical operations
        self.symbolic_processor = SymbolicProcessor()
        
        # Extend supported visualization types
        additional_types = {
            "derivative": self._plot_function_with_derivative,
            "critical_points": self._plot_function_with_critical_points,
            "integral": self._plot_function_with_integral,
            "taylor_series": self._plot_function_with_taylor,
            "vector_field": self._plot_vector_field
        }
        
        # Update supported types
        self.supported_types.update(additional_types)
    
    def _plot_function_with_derivative(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a function and its derivative."""
        try:
            # Extract parameters
            expression = parameters.get("expression")
            if expression is None:
                return {"success": False, "error": "Missing required parameter: expression"}
            
            variable = parameters.get("variable", 'x')
            x_range = parameters.get("x_range", (-10, 10))
            num_points = parameters.get("num_points", 1000)
            
            # Convert expression to SymPy if it's a string
            if isinstance(expression, str):
                x = sp.Symbol(variable)
                expr = sp.sympify(expression)
            else:
                expr = expression
                x = sp.Symbol(variable)
            
            # Compute the derivative
            try:
                derivative = self.symbolic_processor.differentiate(expr, x)
            except Exception as e:
                return {"success": False, "error": f"Failed to compute derivative: {str(e)}"}
            
            # Create labels and prepare for plotting
            labels = [f"f({variable})", f"f'({variable})"]
            functions = [expr, derivative]
            
            # Use multiple functions plotting
            return self._plot_multiple_2d_functions({
                "expressions": functions,
                "labels": labels,
                "x_range": x_range,
                "num_points": num_points,
                "title": f"Function and Derivative of f({variable}) = {sp.latex(expr)}",
                "x_label": variable,
                "y_label": "y",
                "save": parameters.get("save", True),
                "filename": parameters.get("filename")
            })
            
        except Exception as e:
            return {"success": False, "error": f"Error plotting function with derivative: {str(e)}"}
    
    def _plot_function_with_critical_points(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a function with its critical points highlighted."""
        try:
            # Extract parameters
            expression = parameters.get("expression")
            if expression is None:
                return {"success": False, "error": "Missing required parameter: expression"}
            
            variable = parameters.get("variable", 'x')
            x_range = parameters.get("x_range", (-10, 10))
            num_points = parameters.get("num_points", 1000)
            
            # Convert expression to SymPy if it's a string
            if isinstance(expression, str):
                x = sp.Symbol(variable)
                expr = sp.sympify(expression)
            else:
                expr = expression
                x = sp.Symbol(variable)
            
            # Compute the derivative and find critical points
            try:
                derivative = self.symbolic_processor.differentiate(expr, x)
                critical_points = self.symbolic_processor.solve_equation(derivative, x)
                
                # Filter critical points to include only those in the range
                valid_critical_points = []
                for point in critical_points:
                    if isinstance(point, sp.Number):
                        point_value = float(point)
                        if x_range[0] <= point_value <= x_range[1]:
                            valid_critical_points.append(point_value)
                    
                # Compute y-values for critical points
                critical_y_values = []
                f = sp.lambdify(x, expr, "numpy")
                for point in valid_critical_points:
                    try:
                        y_value = float(f(point))
                        if np.isfinite(y_value):
                            critical_y_values.append(y_value)
                        else:
                            valid_critical_points.remove(point)
                    except Exception:
                        valid_critical_points.remove(point)
                
            except Exception as e:
                return {"success": False, "error": f"Failed to compute critical points: {str(e)}"}
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"critical_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Create the basic plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], num_points)
            
            # Convert SymPy expression to NumPy function
            f = sp.lambdify(x, expr, "numpy")
            
            # Calculate y values
            try:
                y_vals = f(x_vals)
                
                # Check for infinities or NaN values
                mask = np.isfinite(y_vals)
                
                # Plot the function
                ax.plot(x_vals[mask], y_vals[mask], label=f"f({variable})")
                
                # Plot critical points if any
                if valid_critical_points:
                    ax.scatter(valid_critical_points, critical_y_values, color='red', s=50, 
                              label="Critical Points", zorder=5)
                    
                    # Add annotations for critical points
                    for i, (x_val, y_val) in enumerate(zip(valid_critical_points, critical_y_values)):
                        ax.annotate(f"({x_val:.2f}, {y_val:.2f})", 
                                   (x_val, y_val),
                                   textcoords="offset points",
                                   xytext=(0, 10),
                                   ha='center')
                
                # Add grid and labels
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(variable)
                ax.set_ylabel("y")
                
                # Add title
                ax.set_title(f"Function with Critical Points: f({variable}) = {sp.latex(expr)}")
                
                # Add legend
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
                        "plot_type": "critical_points",
                        "file_path": save_path,
                        "data": {
                            "expression": str(expr),
                            "expression_latex": sp.latex(expr),
                            "x_range": x_range,
                            "critical_points": [float(p) for p in valid_critical_points],
                            "critical_values": [float(v) for v in critical_y_values]
                        }
                    }
                else:
                    # Convert to base64
                    import io
                    import base64
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    plt.close(fig)
                    
                    buffer.seek(0)
                    image_png = buffer.getvalue()
                    buffer.close()
                    
                    image_base64 = base64.b64encode(image_png).decode('utf-8')
                    
                    return {
                        "success": True,
                        "plot_type": "critical_points",
                        "base64_image": image_base64,
                        "data": {
                            "expression": str(expr),
                            "expression_latex": sp.latex(expr),
                            "x_range": x_range,
                            "critical_points": [float(p) for p in valid_critical_points],
                            "critical_values": [float(v) for v in critical_y_values]
                        }
                    }
                
            except Exception as e:
                return {"success": False, "error": f"Error plotting function: {str(e)}"}
            
        except Exception as e:
            return {"success": False, "error": f"Error plotting function with critical points: {str(e)}"}
    
    def _plot_function_with_integral(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a function with its integral region highlighted."""
        try:
            # Extract parameters
            expression = parameters.get("expression")
            if expression is None:
                return {"success": False, "error": "Missing required parameter: expression"}
            
            # Get integration limits
            lower_bound = parameters.get("lower_bound")
            upper_bound = parameters.get("upper_bound")
            
            if lower_bound is None or upper_bound is None:
                return {"success": False, "error": "Missing required parameters: lower_bound and upper_bound"}
            
            variable = parameters.get("variable", 'x')
            num_points = parameters.get("num_points", 1000)
            
            # Convert expression to SymPy if it's a string
            if isinstance(expression, str):
                x = sp.Symbol(variable)
                expr = sp.sympify(expression)
            else:
                expr = expression
                x = sp.Symbol(variable)
            
            # Ensure bounds are float values
            try:
                lower = float(lower_bound)
                upper = float(upper_bound)
            except ValueError:
                return {"success": False, "error": "Bounds must be numeric values"}
            
            # Compute the definite integral
            try:
                integral_value = self.symbolic_processor.integrate(expr, x, lower, upper)
            except Exception as e:
                return {"success": False, "error": f"Failed to compute integral: {str(e)}"}
            
            # Determine an appropriate x range, including the bounds
            padding = 0.2 * (upper - lower)
            x_range = (lower - padding, upper + padding)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"integral_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Create the plot
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], num_points)
            
            # Convert SymPy expression to NumPy function
            f = sp.lambdify(x, expr, "numpy")
            
            # Calculate y values
            try:
                y_vals = f(x_vals)
                
                # Check for infinities or NaN values
                mask = np.isfinite(y_vals)
                
                # Plot the function
                ax.plot(x_vals[mask], y_vals[mask], label=f"f({variable})")
                
                # Generate points for the integral region
                x_integral = np.linspace(lower, upper, 100)
                y_integral = f(x_integral)
                
                # Create vertices for polygon
                verts = [(lower, 0)] + list(zip(x_integral, y_integral)) + [(upper, 0)]
                poly = Polygon(verts, facecolor='0.9', edgecolor='0.5', alpha=0.5)
                ax.add_patch(poly)
                
                # Calculate the approximate integral value using NumPy
                try:
                    numerical_integral = np.trapz(y_integral, x_integral)
                except Exception:
                    numerical_integral = None
                
                # Add text about the integral value
                text = f"∫({lower:.1f})^({upper:.1f}) f({variable}) d{variable} = {float(integral_value):.4f}"
                ax.text(0.95, 0.95, text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Draw lines at the bounds
                ax.axvline(x=lower, color='r', linestyle='--', alpha=0.5)
                ax.axvline(x=upper, color='r', linestyle='--', alpha=0.5)
                
                # Add grid and labels
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(variable)
                ax.set_ylabel("y")
                
                # Add title
                ax.set_title(f"Integral of f({variable}) = {sp.latex(expr)}")
                
                # Save or encode the figure
                if save_path:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                    
                    # Save the figure
                    plt.savefig(save_path)
                    plt.close(fig)
                    
                    return {
                        "success": True,
                        "plot_type": "integral",
                        "file_path": save_path,
                        "data": {
                            "expression": str(expr),
                            "expression_latex": sp.latex(expr),
                            "lower_bound": lower,
                            "upper_bound": upper,
                            "integral_value": float(integral_value),
                            "numerical_integral": float(numerical_integral) if numerical_integral is not None else None
                        }
                    }
                else:
                    # Convert to base64
                    import io
                    import base64
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    plt.close(fig)
                    
                    buffer.seek(0)
                    image_png = buffer.getvalue()
                    buffer.close()
                    
                    image_base64 = base64.b64encode(image_png).decode('utf-8')
                    
                    return {
                        "success": True,
                        "plot_type": "integral",
                        "base64_image": image_base64,
                        "data": {
                            "expression": str(expr),
                            "expression_latex": sp.latex(expr),
                            "lower_bound": lower,
                            "upper_bound": upper,
                            "integral_value": float(integral_value),
                            "numerical_integral": float(numerical_integral) if numerical_integral is not None else None
                        }
                    }
                
            except Exception as e:
                return {"success": False, "error": f"Error plotting function with integral: {str(e)}"}
            
        except Exception as e:
            return {"success": False, "error": f"Error in integral visualization: {str(e)}"}
    
    def _plot_function_with_taylor(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a function with its Taylor series approximations."""
        try:
            # Extract parameters
            expression = parameters.get("expression")
            if expression is None:
                return {"success": False, "error": "Missing required parameter: expression"}
            
            variable = parameters.get("variable", 'x')
            x_range = parameters.get("x_range", (-5, 5))
            center = parameters.get("center", 0)
            orders = parameters.get("orders", [1, 3, 5])
            num_points = parameters.get("num_points", 1000)
            
            # Convert expression to SymPy if it's a string
            if isinstance(expression, str):
                x = sp.Symbol(variable)
                expr = sp.sympify(expression)
            else:
                expr = expression
                x = sp.Symbol(variable)
            
            # Compute Taylor series approximations
            taylor_series = []
            labels = [f"f({variable})"]
            functions = [expr]
            
            try:
                for order in orders:
                    # Compute Taylor series using SymPy's series function
                    series_expr = expr.series(x, center, order + 1).removeO()
                    taylor_series.append(series_expr)
                    functions.append(series_expr)
                    labels.append(f"Order {order}")
            except Exception as e:
                return {"success": False, "error": f"Failed to compute Taylor series: {str(e)}"}
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"taylor_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Use multiple functions plotting
            return self._plot_multiple_2d_functions({
                "expressions": functions,
                "labels": labels,
                "x_range": x_range,
                "num_points": num_points,
                "title": f"Taylor Series Approximations of f({variable}) = {sp.latex(expr)}",
                "x_label": variable,
                "y_label": "y",
                "save": parameters.get("save", True),
                "filename": filename if filename else None
            })
            
        except Exception as e:
            return {"success": False, "error": f"Error in Taylor series visualization: {str(e)}"}
    
    def _plot_vector_field(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a 2D vector field."""
        try:
            # Extract parameters
            x_component = parameters.get("x_component")
            y_component = parameters.get("y_component")
            
            if x_component is None or y_component is None:
                return {"success": False, "error": "Missing required parameters: x_component and y_component"}
            
            x_range = parameters.get("x_range", (-5, 5))
            y_range = parameters.get("y_range", (-5, 5))
            grid_density = parameters.get("grid_density", 20)
            title = parameters.get("title")
            
            # Convert expressions to SymPy if they're strings
            if isinstance(x_component, str) and isinstance(y_component, str):
                x, y = sp.symbols('x y')
                x_expr = sp.sympify(x_component)
                y_expr = sp.sympify(y_component)
            else:
                x_expr = x_component
                y_expr = y_component
                x, y = sp.symbols('x y')
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"vector_field_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Create the plot
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Generate grid points
            x_grid = np.linspace(x_range[0], x_range[1], grid_density)
            y_grid = np.linspace(y_range[0], y_range[1], grid_density)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Convert SymPy expressions to NumPy functions
            f_x = sp.lambdify((x, y), x_expr, "numpy")
            f_y = sp.lambdify((x, y), y_expr, "numpy")
            
            # Calculate vector components
            try:
                U = f_x(X, Y)
                V = f_y(X, Y)
                
                # Check for infinities or NaN values
                mask = np.isfinite(U) & np.isfinite(V)
                
                if not np.any(mask):
                    return {"success": False, "error": "No finite values in the vector field"}
                
                # Replace NaN and infinity values with 0
                U = np.where(np.isfinite(U), U, 0)
                V = np.where(np.isfinite(V), V, 0)
                
                # Normalize if specified (optional)
                if parameters.get("normalize", False):
                    norm = np.sqrt(U**2 + V**2)
                    # Avoid division by zero
                    mask = norm > 0
                    U_norm = np.zeros_like(U)
                    V_norm = np.zeros_like(V)
                    U_norm[mask] = U[mask] / norm[mask]
                    V_norm[mask] = V[mask] / norm[mask]
                    U, V = U_norm, V_norm
                
                # Plot the vector field
                ax.quiver(X, Y, U, V, pivot='mid', color='blue')
                
                # Add grid and labels
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
                # Set equal aspect ratio
                ax.set_aspect('equal')
                
                # Add title
                if title:
                    ax.set_title(title)
                else:
                    ax.set_title(f"Vector Field: ({sp.latex(x_expr)}, {sp.latex(y_expr)})")
                
                # Save or encode the figure
                if save_path:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                    
                    # Save the figure
                    plt.savefig(save_path)
                    plt.close(fig)
                    
                    return {
                        "success": True,
                        "plot_type": "vector_field",
                        "file_path": save_path,
                        "data": {
                            "x_component": str(x_expr),
                            "y_component": str(y_expr),
                            "x_range": x_range,
                            "y_range": y_range
                        }
                    }
                else:
                    # Convert to base64
                    import io
                    import base64
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    plt.close(fig)
                    
                    buffer.seek(0)
                    image_png = buffer.getvalue()
                    buffer.close()
                    
                    image_base64 = base64.b64encode(image_png).decode('utf-8')
                    
                    return {
                        "success": True,
                        "plot_type": "vector_field",
                        "base64_image": image_base64,
                        "data": {
                            "x_component": str(x_expr),
                            "y_component": str(y_expr),
                            "x_range": x_range,
                            "y_range": y_range
                        }
                    }
                
            except Exception as e:
                return {"success": False, "error": f"Error calculating vector field: {str(e)}"}
            
        except Exception as e:
            return {"success": False, "error": f"Error in vector field visualization: {str(e)}"}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return the agent's capabilities.
        
        Returns:
            Dictionary of agent capabilities
        """
        capabilities = super().get_capabilities()
        capabilities["agent_type"] = "advanced_visualization"
        capabilities["supported_types"] = list(self.supported_types.keys())
        capabilities["advanced_features"] = ["derivatives", "integrals", "taylor_series", "critical_points", "vector_fields"]
        
        return capabilities

import os
import json
import sympy as sp
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import uuid
from datetime import datetime
import base64

# Local imports
from visualization.plotting.plot_2d import plot_function_2d, plot_multiple_functions_2d
from visualization.plotting.plot_3d import plot_function_3d, plot_parametric_3d
from visualization.plotting.statistical import plot_histogram, plot_scatter
from database.access.visualization_repository import VisualizationRepository

class VisualizationAgent:
    """
    Agent responsible for generating mathematical visualizations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Visualization Agent.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        self.config = config or {}
        
        # Set default values from config or use defaults
        self.storage_dir = self.config.get("storage_dir", "visualizations")
        self.default_format = self.config.get("default_format", "png")
        self.default_dpi = self.config.get("default_dpi", 100)
        self.max_width = self.config.get("max_width", 1200)
        self.max_height = self.config.get("max_height", 800)
        
        # Initialize storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize database connection if available
        self.db_repository = None
        if self.config.get("use_database", True):
            try:
                self.db_repository = VisualizationRepository()
            except Exception as e:
                print(f"Warning: Failed to initialize database repository: {e}")
        
        # Register supported visualization types
        self.supported_types = {
            "function_2d": self._plot_2d_function,
            "functions_2d": self._plot_multiple_2d_functions,
            "function_3d": self._plot_3d_function,
            "parametric_3d": self._plot_parametric_3d,
            "histogram": self._plot_histogram,
            "scatter": self._plot_scatter
        }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a visualization request message.
        
        Args:
            message: The message containing visualization request
            
        Returns:
            Response with visualization results
        """
        try:
            # Extract message information
            body = message.get("body", {})
            visualization_type = body.get("visualization_type")
            parameters = body.get("parameters", {})
            
            # Check if visualization type is supported
            if visualization_type not in self.supported_types:
                return {
                    "success": False,
                    "error": f"Unsupported visualization type: {visualization_type}",
                    "supported_types": list(self.supported_types.keys())
                }
            
            # Call the appropriate visualization function
            visualization_func = self.supported_types[visualization_type]
            result = visualization_func(parameters)
            
            # Store visualization in database if successful and repository exists
            if result.get("success", False) and self.db_repository:
                try:
                    # Store only if we have a file path
                    if "file_path" in result:
                        viz_id = self.db_repository.store_visualization(
                            visualization_type=visualization_type,
                            parameters=parameters,
                            file_path=result["file_path"],
                            metadata=result.get("data", {})
                        )
                        result["visualization_id"] = viz_id
                except Exception as e:
                    # Don't fail if database storage fails
                    result["warning"] = f"Visualization created but failed to store in database: {e}"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process visualization request: {e}"
            }
    
    def _plot_2d_function(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a 2D function."""
        try:
            # Extract parameters
            expression = parameters.get("expression")
            if expression is None:
                return {"success": False, "error": "Missing required parameter: expression"}
                
            # Handle optional parameters
            x_range = parameters.get("x_range", (-10, 10))
            num_points = parameters.get("num_points", 1000)
            title = parameters.get("title")
            x_label = parameters.get("x_label", "x")
            y_label = parameters.get("y_label", "y")
            show_grid = parameters.get("show_grid", True)
            figsize = parameters.get("figsize", (8, 6))
            dpi = parameters.get("dpi", self.default_dpi)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"function_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_function_2d(
                function_expr=expression,
                x_range=x_range,
                num_points=num_points,
                title=title,
                x_label=x_label,
                y_label=y_label,
                show_grid=show_grid,
                figsize=figsize,
                dpi=dpi,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in 2D function plotting: {str(e)}"}
    
    def _plot_multiple_2d_functions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot multiple 2D functions on the same axes."""
        try:
            # Extract parameters
            expressions = parameters.get("expressions")
            if not expressions:
                return {"success": False, "error": "Missing required parameter: expressions"}
            
            # Ensure expressions is a list
            if not isinstance(expressions, list):
                expressions = [expressions]
                
            # Handle optional parameters
            labels = parameters.get("labels")
            x_range = parameters.get("x_range", (-10, 10))
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
            
            # Generate the plot
            result = plot_multiple_functions_2d(
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
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False, 
                "error": f"Error in multiple 2D function plotting: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _plot_3d_function(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a 3D function."""
        try:
            # Extract parameters
            expression = parameters.get("expression")
            if expression is None:
                return {"success": False, "error": "Missing required parameter: expression"}
                
            # Handle optional parameters
            x_range = parameters.get("x_range", (-5, 5))
            y_range = parameters.get("y_range", (-5, 5))
            num_points = parameters.get("num_points", 50)
            title = parameters.get("title")
            x_label = parameters.get("x_label", "x")
            y_label = parameters.get("y_label", "y")
            z_label = parameters.get("z_label", "z")
            figsize = parameters.get("figsize", (10, 8))
            cmap = parameters.get("cmap", "viridis")
            view_angle = parameters.get("view_angle", (30, 30))
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"function_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_function_3d(
                function_expr=expression,
                x_range=x_range,
                y_range=y_range,
                num_points=num_points,
                title=title,
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
                figsize=figsize,
                cmap=cmap,
                view_angle=view_angle,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in 3D function plotting: {str(e)}"}
    
    def _plot_parametric_3d(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a 3D parametric curve."""
        try:
            # Extract parameters
            x_expr = parameters.get("x_expression")
            y_expr = parameters.get("y_expression")
            z_expr = parameters.get("z_expression")
            
            if not all([x_expr, y_expr, z_expr]):
                return {"success": False, "error": "Missing required parametric expressions"}
                
            # Handle optional parameters
            t_range = parameters.get("t_range", (0, 2*np.pi))
            num_points = parameters.get("num_points", 1000)
            title = parameters.get("title")
            x_label = parameters.get("x_label", "x")
            y_label = parameters.get("y_label", "y")
            z_label = parameters.get("z_label", "z")
            figsize = parameters.get("figsize", (10, 8))
            color = parameters.get("color", "blue")
            view_angle = parameters.get("view_angle", (30, 30))
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"parametric_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_parametric_3d(
                x_expr=x_expr,
                y_expr=y_expr,
                z_expr=z_expr,
                t_range=t_range,
                num_points=num_points,
                title=title,
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
                figsize=figsize,
                color=color,
                view_angle=view_angle,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in 3D parametric plotting: {str(e)}"}
    
    def _plot_histogram(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a histogram."""
        try:
            # Extract parameters
            data = parameters.get("data")
            if data is None:
                return {"success": False, "error": "Missing required parameter: data"}
                
            # Handle optional parameters
            bins = parameters.get("bins", 'auto')
            title = parameters.get("title")
            x_label = parameters.get("x_label", "Value")
            y_label = parameters.get("y_label", "Frequency")
            figsize = parameters.get("figsize", (8, 6))
            color = parameters.get("color", '#1f77b4')
            edgecolor = parameters.get("edgecolor", 'black')
            density = parameters.get("density", False)
            show_kde = parameters.get("show_kde", False)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_histogram(
                data=data,
                bins=bins,
                title=title,
                x_label=x_label,
                y_label=y_label,
                figsize=figsize,
                color=color,
                edgecolor=edgecolor,
                density=density,
                show_kde=show_kde,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in histogram plotting: {str(e)}"}
    
    def _plot_scatter(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a scatter plot."""
        try:
            # Extract parameters
            x_data = parameters.get("x_data")
            y_data = parameters.get("y_data")
            
            if x_data is None or y_data is None:
                return {"success": False, "error": "Missing required parameters: x_data and y_data"}
                
            # Handle optional parameters
            title = parameters.get("title")
            x_label = parameters.get("x_label", "X")
            y_label = parameters.get("y_label", "Y")
            figsize = parameters.get("figsize", (8, 6))
            color = parameters.get("color", '#1f77b4')
            alpha = parameters.get("alpha", 0.7)
            show_regression = parameters.get("show_regression", False)
            marker = parameters.get("marker", 'o')
            size = parameters.get("size", 30)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_scatter(
                x_data=x_data,
                y_data=y_data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                figsize=figsize,
                color=color,
                alpha=alpha,
                show_regression=show_regression,
                marker=marker,
                size=size,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in scatter plotting: {str(e)}"}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return the agent's capabilities.
        
        Returns:
            Dictionary of agent capabilities
        """
        return {
            "agent_type": "visualization",
            "supported_types": list(self.supported_types.keys()),
            "max_dimensions": {
                "width": self.max_width,
                "height": self.max_height
            },
            "supported_formats": [self.default_format, "svg", "pdf"]
        }

    def generate_visualization(
        self, 
        expression: Union[str, List[str]], 
        viz_type: str = "plot",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a visualization based on the expression and type.
        
        Args:
            expression: Mathematical expression or list of expressions to visualize
            viz_type: Type of visualization ('plot', 'multi_plot', 'surface', etc.)
            parameters: Additional parameters for the visualization
            
        Returns:
            Dictionary with visualization results including base64 image data
        """
        if parameters is None:
            parameters = {}
            
        try:
            # Map the viz_type to the internal visualization types
            viz_type_mapping = {
                "plot": "function_2d",
                "multi_plot": "functions_2d",
                "surface": "function_3d",
                "parametric": "parametric_3d",
                "histogram": "histogram",
                "scatter": "scatter"
            }
            
            internal_type = viz_type_mapping.get(viz_type)
            if not internal_type:
                return {
                    "success": False,
                    "error": f"Unsupported visualization type: {viz_type}",
                    "supported_types": list(viz_type_mapping.keys())
                }
                
            # Prepare parameters based on visualization type
            if internal_type == "function_2d":
                # Single 2D function
                viz_params = {
                    "expression": expression,
                    **parameters
                }
                result = self._plot_2d_function(viz_params)
                
            elif internal_type == "functions_2d":
                # Multiple 2D functions
                viz_params = {
                    "expressions": expression if isinstance(expression, list) else [expression],
                    **parameters
                }
                result = self._plot_multiple_2d_functions(viz_params)
                
            elif internal_type == "function_3d":
                # 3D surface plot
                viz_params = {
                    "expression": expression,
                    **parameters
                }
                result = self._plot_3d_function(viz_params)
                
            elif internal_type == "parametric_3d":
                # 3D parametric plot
                viz_params = {
                    "expression": expression,
                    **parameters
                }
                result = self._plot_parametric_3d(viz_params)
                
            elif internal_type == "histogram":
                # Histogram
                viz_params = {
                    "data": expression,
                    **parameters
                }
                result = self._plot_histogram(viz_params)
                
            elif internal_type == "scatter":
                # Scatter plot
                viz_params = {
                    "data": expression,
                    **parameters
                }
                result = self._plot_scatter(viz_params)
            
            # Process the result to ensure it contains image_data
            if result.get("success", False):
                # If we have base64 image, use it as image_data
                if "base64_image" in result:
                    result["image_data"] = result["base64_image"]
                    
                # If we have a file path, read the file and encode as base64
                elif "file_path" in result and os.path.exists(result["file_path"]):
                    with open(result["file_path"], "rb") as f:
                        file_data = f.read()
                        result["image_data"] = base64.b64encode(file_data).decode('utf-8')
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": f"Failed to generate visualization: {str(e)}",
                "traceback": traceback.format_exc()
            }

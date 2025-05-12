"""
Interactive Visualization Agent for the Mathematical Multimodal LLM System.

This module provides an agent specialized in creating interactive visualizations
using Plotly and other interactive technologies.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import sympy as sp
import uuid
from datetime import datetime

from visualization.agent.viz_agent import VisualizationAgent
from visualization.plotting.interactive import (
    interactive_function_2d,
    interactive_function_3d,
    interactive_scatter_3d,
    interactive_multivariate_function
)

class InteractiveVisualizationAgent(VisualizationAgent):
    """
    Agent responsible for generating interactive mathematical visualizations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Interactive Visualization Agent.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        # Initialize base agent
        super().__init__(config)
        
        # Set default values specific to interactive visualizations
        self.html_dir = self.config.get("html_dir", "visualizations/interactive")
        
        # Create HTML directory if it doesn't exist
        os.makedirs(self.html_dir, exist_ok=True)
        
        # Register interactive visualization types
        interactive_types = {
            "interactive_function_2d": self._interactive_2d_function,
            "interactive_function_3d": self._interactive_3d_function,
            "interactive_scatter_3d": self._interactive_scatter_3d,
            "interactive_multivariate": self._interactive_multivariate_function
        }
        
        # Update supported types
        self.supported_types.update(interactive_types)
    
    def _interactive_2d_function(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create an interactive 2D function plot."""
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
            width = parameters.get("width", 800)
            height = parameters.get("height", 500)
            template = parameters.get("template", "plotly_white")
            line_color = parameters.get("line_color", "#1f77b4")
            include_html = parameters.get("include_html", True)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"interactive_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.html"
                save_path = os.path.join(self.html_dir, filename)
            
            # Generate the plot
            result = interactive_function_2d(
                function_expr=expression,
                x_range=x_range,
                num_points=num_points,
                title=title,
                x_label=x_label,
                y_label=y_label,
                show_grid=show_grid,
                width=width,
                height=height,
                template=template,
                line_color=line_color,
                save_path=save_path,
                include_html=include_html
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in interactive 2D function plotting: {str(e)}"}
    
    def _interactive_3d_function(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create an interactive 3D function plot."""
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
            width = parameters.get("width", 800)
            height = parameters.get("height", 600)
            template = parameters.get("template", "plotly_white")
            colorscale = parameters.get("colorscale", "viridis")
            include_html = parameters.get("include_html", True)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"interactive_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.html"
                save_path = os.path.join(self.html_dir, filename)
            
            # Generate the plot
            result = interactive_function_3d(
                function_expr=expression,
                x_range=x_range,
                y_range=y_range,
                num_points=num_points,
                title=title,
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
                width=width,
                height=height,
                template=template,
                colorscale=colorscale,
                save_path=save_path,
                include_html=include_html
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in interactive 3D function plotting: {str(e)}"}
    
    def _interactive_scatter_3d(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create an interactive 3D scatter plot."""
        try:
            # Extract parameters
            x_data = parameters.get("x_data")
            y_data = parameters.get("y_data")
            z_data = parameters.get("z_data")
            
            if not all([x_data, y_data, z_data]):
                return {"success": False, "error": "Missing required parameters: x_data, y_data, z_data"}
                
            # Handle optional parameters
            labels = parameters.get("labels")
            title = parameters.get("title")
            x_label = parameters.get("x_label", "X")
            y_label = parameters.get("y_label", "Y")
            z_label = parameters.get("z_label", "Z")
            width = parameters.get("width", 800)
            height = parameters.get("height", 600)
            template = parameters.get("template", "plotly_white")
            colorscale = parameters.get("colorscale", "viridis")
            marker_size = parameters.get("marker_size", 5)
            include_html = parameters.get("include_html", True)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"interactive_scatter_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.html"
                save_path = os.path.join(self.html_dir, filename)
            
            # Generate the plot
            result = interactive_scatter_3d(
                x_data=x_data,
                y_data=y_data,
                z_data=z_data,
                labels=labels,
                title=title,
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
                width=width,
                height=height,
                template=template,
                colorscale=colorscale,
                marker_size=marker_size,
                save_path=save_path,
                include_html=include_html
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in interactive 3D scatter plotting: {str(e)}"}
    
    def _interactive_multivariate_function(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create an interactive multivariate function plot with sliders."""
        try:
            # Extract parameters
            expression = parameters.get("expression")
            sliders = parameters.get("sliders")
            
            if expression is None:
                return {"success": False, "error": "Missing required parameter: expression"}
                
            if sliders is None:
                return {"success": False, "error": "Missing required parameter: sliders"}
                
            # Handle optional parameters
            variable_x = parameters.get("variable_x", "x")
            x_range = parameters.get("x_range", (-10, 10))
            num_points = parameters.get("num_points", 1000)
            title = parameters.get("title")
            x_label = parameters.get("x_label")
            y_label = parameters.get("y_label", "f(x)")
            width = parameters.get("width", 900)
            height = parameters.get("height", 600)
            template = parameters.get("template", "plotly_white")
            line_color = parameters.get("line_color", "#1f77b4")
            include_html = parameters.get("include_html", True)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"interactive_multivariate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.html"
                save_path = os.path.join(self.html_dir, filename)
            
            # Generate the plot
            result = interactive_multivariate_function(
                function_expr=expression,
                sliders=sliders,
                variable_x=variable_x,
                x_range=x_range,
                num_points=num_points,
                title=title,
                x_label=x_label,
                y_label=y_label,
                width=width,
                height=height,
                template=template,
                line_color=line_color,
                save_path=save_path,
                include_html=include_html
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in interactive multivariate function plotting: {str(e)}"}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return the agent's capabilities.
        
        Returns:
            Dictionary of agent capabilities
        """
        capabilities = super().get_capabilities()
        capabilities["agent_type"] = "interactive_visualization"
        capabilities["supported_types"] = list(self.supported_types.keys())
        capabilities["interactive_features"] = [
            "2d_function_plot", 
            "3d_function_plot", 
            "3d_scatter_plot", 
            "multivariate_function_plot"
        ]
        
        return capabilities

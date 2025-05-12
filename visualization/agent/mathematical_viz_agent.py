"""
Mathematical Visualization Agent for the Mathematical Multimodal LLM System.

This module provides an agent specialized in creating visualizations for specific 
mathematical concepts, such as complex numbers, linear transformations, etc.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import sympy as sp
import uuid
from datetime import datetime

from visualization.agent.viz_agent import VisualizationAgent
from visualization.plotting.mathematical import (
    plot_complex_numbers,
    plot_vector_addition_2d,
    plot_linear_transformation_2d,
    plot_probability_distribution
)

class MathematicalVisualizationAgent(VisualizationAgent):
    """
    Agent responsible for generating specialized mathematical visualizations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Mathematical Visualization Agent.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        # Initialize base agent
        super().__init__(config)
        
        # Register specialized visualization types
        specialized_types = {
            "complex_numbers": self._plot_complex_numbers,
            "vector_addition": self._plot_vector_addition,
            "linear_transformation": self._plot_linear_transformation,
            "probability_distribution": self._plot_probability_distribution
        }
        
        # Update supported types
        self.supported_types.update(specialized_types)
    
    def _plot_complex_numbers(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot complex numbers in the complex plane."""
        try:
            # Extract parameters
            complex_numbers = parameters.get("complex_numbers")
            if complex_numbers is None:
                return {"success": False, "error": "Missing required parameter: complex_numbers"}
                
            # Convert string representations to complex if necessary
            parsed_numbers = []
            for z in complex_numbers:
                if isinstance(z, (int, float, complex)):
                    parsed_numbers.append(z)
                elif isinstance(z, str):
                    try:
                        # Handle strings like "1+2j" or expressions like "1+2i"
                        z = z.replace('i', 'j')
                        parsed_numbers.append(complex(z))
                    except ValueError:
                        return {"success": False, "error": f"Invalid complex number format: {z}"}
                else:
                    return {"success": False, "error": f"Unsupported complex number format: {z}"}
            
            # Handle optional parameters
            labels = parameters.get("labels")
            title = parameters.get("title", "Complex Numbers")
            show_unit_circle = parameters.get("show_unit_circle", True)
            show_real_axis = parameters.get("show_real_axis", True)
            show_imag_axis = parameters.get("show_imag_axis", True)
            show_abs_value = parameters.get("show_abs_value", False)
            show_phase = parameters.get("show_phase", False)
            figsize = parameters.get("figsize", (8, 8))
            dpi = parameters.get("dpi", self.default_dpi)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"complex_numbers_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_complex_numbers(
                complex_numbers=parsed_numbers,
                labels=labels,
                title=title,
                show_unit_circle=show_unit_circle,
                show_real_axis=show_real_axis,
                show_imag_axis=show_imag_axis,
                show_abs_value=show_abs_value,
                show_phase=show_phase,
                figsize=figsize,
                dpi=dpi,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in complex numbers plotting: {str(e)}"}
    
    def _plot_vector_addition(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot vector addition in 2D space."""
        try:
            # Extract parameters
            vectors = parameters.get("vectors")
            if vectors is None:
                return {"success": False, "error": "Missing required parameter: vectors"}
                
            # Validate vectors
            for v in vectors:
                if not isinstance(v, (list, tuple)) or len(v) != 2:
                    return {"success": False, "error": f"Invalid vector format: {v}. Expected (x, y) tuple."}
            
            # Handle optional parameters
            labels = parameters.get("labels")
            title = parameters.get("title", "Vector Addition")
            origin = parameters.get("origin", (0, 0))
            show_resultant = parameters.get("show_resultant", True)
            colors = parameters.get("colors")
            sequential = parameters.get("sequential", False)
            figsize = parameters.get("figsize", (8, 8))
            dpi = parameters.get("dpi", self.default_dpi)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"vector_addition_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_vector_addition_2d(
                vectors=vectors,
                labels=labels,
                title=title,
                origin=origin,
                show_resultant=show_resultant,
                colors=colors,
                sequential=sequential,
                figsize=figsize,
                dpi=dpi,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in vector addition plotting: {str(e)}"}
    
    def _plot_linear_transformation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a 2D linear transformation."""
        try:
            # Extract parameters
            transformation_matrix = parameters.get("transformation_matrix")
            if transformation_matrix is None:
                return {"success": False, "error": "Missing required parameter: transformation_matrix"}
                
            # Validate matrix
            if (not isinstance(transformation_matrix, (list, tuple)) or 
                len(transformation_matrix) != 2 or 
                not all(len(row) == 2 for row in transformation_matrix)):
                return {"success": False, "error": "Transformation matrix must be a 2x2 matrix"}
            
            # Handle optional parameters
            grid_lines = parameters.get("grid_lines", 10)
            grid_range = parameters.get("grid_range", 4.0)
            title = parameters.get("title", "Linear Transformation")
            show_basis_vectors = parameters.get("show_basis_vectors", True)
            figsize = parameters.get("figsize", (12, 6))
            dpi = parameters.get("dpi", self.default_dpi)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"linear_transformation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_linear_transformation_2d(
                transformation_matrix=transformation_matrix,
                grid_lines=grid_lines,
                grid_range=grid_range,
                title=title,
                show_basis_vectors=show_basis_vectors,
                figsize=figsize,
                dpi=dpi,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in linear transformation plotting: {str(e)}"}
    
    def _plot_probability_distribution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a probability distribution."""
        try:
            # Extract parameters
            distribution_type = parameters.get("distribution_type")
            dist_parameters = parameters.get("parameters")
            
            if distribution_type is None:
                return {"success": False, "error": "Missing required parameter: distribution_type"}
                
            if dist_parameters is None:
                return {"success": False, "error": "Missing required parameter: parameters"}
            
            # Handle optional parameters
            x_range = parameters.get("x_range")
            num_points = parameters.get("num_points", 1000)
            title = parameters.get("title")
            show_mean = parameters.get("show_mean", True)
            show_std_dev = parameters.get("show_std_dev", True)
            show_percentiles = parameters.get("show_percentiles", False)
            percentiles = parameters.get("percentiles", [5, 25, 50, 75, 95])
            figsize = parameters.get("figsize", (10, 6))
            dpi = parameters.get("dpi", self.default_dpi)
            
            # Determine output path
            save_path = None
            if parameters.get("save", True):
                filename = parameters.get("filename")
                if not filename:
                    filename = f"probability_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{self.default_format}"
                save_path = os.path.join(self.storage_dir, filename)
            
            # Generate the plot
            result = plot_probability_distribution(
                distribution_type=distribution_type,
                parameters=dist_parameters,
                x_range=x_range,
                num_points=num_points,
                title=title,
                show_mean=show_mean,
                show_std_dev=show_std_dev,
                show_percentiles=show_percentiles,
                percentiles=percentiles,
                figsize=figsize,
                dpi=dpi,
                save_path=save_path
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error in probability distribution plotting: {str(e)}"}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return the agent's capabilities.
        
        Returns:
            Dictionary of agent capabilities
        """
        capabilities = super().get_capabilities()
        capabilities["agent_type"] = "mathematical_visualization"
        capabilities["supported_types"] = list(self.supported_types.keys())
        capabilities["mathematical_domains"] = [
            "complex_analysis", 
            "linear_algebra",
            "probability_theory",
            "vector_calculus"
        ]
        
        return capabilities

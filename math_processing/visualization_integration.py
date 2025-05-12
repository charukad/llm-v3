"""
Visualization Integration for Mathematical Processing Component

This module provides the integration layer between the Mathematical Processing
components and the Visualization components, allowing direct visualization
of mathematical expressions, computation results, and step-by-step solutions.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import sympy as sp
import numpy as np
from sympy.plotting import plot as sp_plot

from math_processing.expressions.latex_parser import parse_latex_to_sympy
from math_processing.expressions.converters import sympy_to_latex
from math_processing.computation.sympy_wrapper import SymbolicProcessor
from orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus

logger = logging.getLogger(__name__)

class VisualizationIntegrator:
    """
    Integrates Mathematical Processing with Visualization components.
    
    This class serves as a bridge between mathematical computations and
    visualizations, allowing direct creation of appropriate visualizations
    for mathematical expressions and results.
    """
    
    def __init__(self, message_bus: Optional[RabbitMQBus] = None):
        """
        Initialize the VisualizationIntegrator.
        
        Args:
            message_bus: Optional message bus for agent communication.
                         If None, a new instance will be created.
        """
        self.message_bus = message_bus or RabbitMQBus()
        self.symbolic_processor = SymbolicProcessor()
        
    def visualize_expression(self, 
                             expression: Union[str, sp.Expr],
                             visualization_type: str = "auto",
                             **kwargs) -> Dict[str, Any]:
        """
        Create visualization for a mathematical expression.
        
        Args:
            expression: Mathematical expression as LaTeX string or SymPy expression
            visualization_type: Type of visualization ("auto", "function_2d", "function_3d", etc.)
            **kwargs: Additional parameters for visualization
            
        Returns:
            Dictionary with visualization metadata and status
        """
        # Convert to SymPy expression if string was provided
        sympy_expr = self._ensure_sympy_expression(expression)
        
        # Determine appropriate visualization if set to auto
        if visualization_type == "auto":
            visualization_type = self._determine_visualization_type(sympy_expr)
            
        # Prepare visualization request
        request = self._prepare_visualization_request(sympy_expr, visualization_type, **kwargs)
        
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def visualize_computation_result(self, 
                                     result: Dict[str, Any],
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for a computation result.
        
        Args:
            result: Computation result from Mathematical Processing Agent
            context: Additional context information
            
        Returns:
            Dictionary with visualization metadata and status
        """
        # Extract relevant information from computation result
        operation = result.get("operation")
        expression = result.get("expression")
        result_expr = result.get("result")
        domain = result.get("domain")
        
        # Handle different operation types
        if operation == "differentiate":
            return self._visualize_derivative(expression, result_expr, context)
        elif operation == "integrate":
            return self._visualize_integral(expression, result_expr, context)
        elif operation == "solve":
            return self._visualize_solution(expression, result_expr, context)
        elif operation == "series":
            return self._visualize_series(expression, result_expr, context)
        elif operation == "limit":
            return self._visualize_limit(expression, result_expr, context)
        else:
            # Default visualization based on domain
            return self._visualize_by_domain(expression, result_expr, domain, context)
    
    def visualize_step_solution(self, 
                               solution_steps: List[Dict[str, Any]],
                               context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Create visualizations for a step-by-step solution.
        
        Args:
            solution_steps: List of solution steps from step generator
            context: Additional context information
            
        Returns:
            List of visualization results, one for each significant step
        """
        visualizations = []
        
        # Identify key steps that would benefit from visualization
        key_steps = self._identify_key_steps(solution_steps)
        
        # Create visualization for each key step
        for step in key_steps:
            step_expr = step.get("output")
            step_type = step.get("operation")
            step_explanation = step.get("explanation")
            
            visualization = self.visualize_expression(
                expression=step_expr,
                visualization_type="auto",
                title=f"Step {step.get('step_number')}: {step_explanation}",
                highlight_changes=True,
                previous_expr=step.get("input") if "input" in step else None,
                step_type=step_type
            )
            
            visualizations.append(visualization)
            
        return visualizations
    
    def _ensure_sympy_expression(self, expression: Union[str, sp.Expr]) -> sp.Expr:
        """
        Ensure the expression is a SymPy expression.
        
        Args:
            expression: Either a LaTeX string or a SymPy expression
            
        Returns:
            SymPy expression
        """
        if isinstance(expression, str):
            # Check if it's a LaTeX string
            if "\\" in expression or "{" in expression:
                return parse_latex_to_sympy(expression)
            else:
                # Simple string expression
                x, y, z = sp.symbols('x y z')
                return sp.sympify(expression)
        elif isinstance(expression, sp.Expr):
            return expression
        else:
            raise ValueError(f"Unsupported expression type: {type(expression)}")
    
    def _determine_visualization_type(self, sympy_expr: sp.Expr) -> str:
        """
        Determine the most appropriate visualization type for an expression.
        
        Args:
            sympy_expr: SymPy expression to visualize
            
        Returns:
            Visualization type string
        """
        free_symbols = sympy_expr.free_symbols
        
        # Check for number of variables
        if len(free_symbols) == 0:
            # Constant expression
            return "text"
        elif len(free_symbols) == 1:
            # Single variable expression - use 2D function plot
            return "function_2d"
        elif len(free_symbols) == 2:
            # Two variables - use 3D surface plot
            return "function_3d"
        else:
            # More than two variables - use parametric 3D or slices
            return "multivariate"
        
    def _prepare_visualization_request(self, 
                                      sympy_expr: sp.Expr, 
                                      visualization_type: str,
                                      **kwargs) -> Dict[str, Any]:
        """
        Prepare a visualization request for the visualization agent.
        
        Args:
            sympy_expr: SymPy expression to visualize
            visualization_type: Type of visualization
            **kwargs: Additional parameters
            
        Returns:
            Request dictionary
        """
        # Convert expression to LaTeX for transport
        latex_expr = sympy_to_latex(sympy_expr)
        
        # Prepare request body
        request = {
            "visualization_type": visualization_type,
            "latex_expression": latex_expr,
            "sympy_expression_str": str(sympy_expr),
            "parameters": kwargs
        }
        
        # Add free symbols information
        free_symbols = list(sympy_expr.free_symbols)
        request["variables"] = [str(symbol) for symbol in free_symbols]
        
        # Add domain information if available
        if "domain" in kwargs:
            request["domain"] = kwargs["domain"]
            
        # Add range information for plotting
        if visualization_type in ["function_2d", "function_3d"] and len(free_symbols) > 0:
            # Default ranges
            if "x_range" not in kwargs and len(free_symbols) >= 1:
                request["x_range"] = [-10, 10]
            if "y_range" not in kwargs and len(free_symbols) >= 2:
                request["y_range"] = [-10, 10]
        
        return request
    
    def _visualize_derivative(self, 
                             expression: Union[str, sp.Expr],
                             derivative: Union[str, sp.Expr],
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for a derivative computation.
        
        Args:
            expression: Original expression
            derivative: Derivative result
            context: Additional context
            
        Returns:
            Visualization result
        """
        expr = self._ensure_sympy_expression(expression)
        deriv = self._ensure_sympy_expression(derivative)
        
        # Prepare a multi-function visualization showing both the original function and its derivative
        request = {
            "visualization_type": "multi_function_2d",
            "functions": [
                {
                    "expression": sympy_to_latex(expr),
                    "label": "f(x)",
                    "color": "blue"
                },
                {
                    "expression": sympy_to_latex(deriv),
                    "label": "f'(x)",
                    "color": "red"
                }
            ],
            "title": "Function and its Derivative",
            "show_grid": True,
            "x_label": "x",
            "y_label": "y"
        }
        
        # Add any critical points
        try:
            x = list(expr.free_symbols)[0]  # Get the main variable
            critical_points = sp.solve(deriv, x)
            
            if critical_points:
                request["annotations"] = []
                for point in critical_points:
                    try:
                        y_val = expr.subs(x, point)
                        request["annotations"].append({
                            "x": float(point),
                            "y": float(y_val),
                            "text": f"Critical point ({float(point):.2f}, {float(y_val):.2f})"
                        })
                    except (TypeError, ValueError):
                        # Skip complex or non-numerical points
                        continue
        except Exception as e:
            logger.warning(f"Could not compute critical points: {e}")
        
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def _visualize_integral(self, 
                           expression: Union[str, sp.Expr],
                           integral_result: Union[str, sp.Expr],
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for an integral computation.
        
        Args:
            expression: Original expression
            integral_result: Integration result
            context: Additional context
            
        Returns:
            Visualization result
        """
        expr = self._ensure_sympy_expression(expression)
        
        # Check if we have bounds for a definite integral
        bounds = context.get("bounds") if context else None
        
        if bounds and len(bounds) == 2:
            # Definite integral - shade the area under the curve
            request = {
                "visualization_type": "integral_2d",
                "expression": sympy_to_latex(expr),
                "lower_bound": bounds[0],
                "upper_bound": bounds[1],
                "title": f"Definite Integral from {bounds[0]} to {bounds[1]}",
                "shade_area": True,
                "show_grid": True
            }
        else:
            # Indefinite integral - show both the original function and its integral
            integral = self._ensure_sympy_expression(integral_result)
            
            request = {
                "visualization_type": "multi_function_2d",
                "functions": [
                    {
                        "expression": sympy_to_latex(expr),
                        "label": "f(x)",
                        "color": "blue"
                    },
                    {
                        "expression": sympy_to_latex(integral),
                        "label": "∫f(x)dx",
                        "color": "green"
                    }
                ],
                "title": "Function and its Integral",
                "show_grid": True
            }
        
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def _visualize_solution(self, 
                           equation: Union[str, sp.Expr],
                           solutions: Union[List, str, sp.Expr],
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for an equation solution.
        
        Args:
            equation: Original equation
            solutions: Solution results
            context: Additional context
            
        Returns:
            Visualization result
        """
        eq_expr = self._ensure_sympy_expression(equation)
        
        # Extract domain information for appropriate visualization
        domain = context.get("domain") if context else None
        
        if domain == "algebra" or domain is None:
            # For algebraic equations, visualize the function and its roots
            # Convert from equation to function by moving everything to left side
            if isinstance(eq_expr, sp.Eq):
                func_expr = eq_expr.lhs - eq_expr.rhs
            else:
                func_expr = eq_expr
                
            # Convert solutions to a list if needed
            if not isinstance(solutions, list):
                sol_list = [solutions]
            else:
                sol_list = solutions
                
            # Ensure all solutions are in sympy format
            sol_exprs = [self._ensure_sympy_expression(sol) if isinstance(sol, str) else sol 
                         for sol in sol_list]
            
            # Prepare visualization request
            request = {
                "visualization_type": "function_roots_2d",
                "expression": sympy_to_latex(func_expr),
                "roots": [float(sol) if sol.is_real else None for sol in sol_exprs],
                "title": "Function and its Roots (Solutions)",
                "show_grid": True,
                "highlight_roots": True
            }
            
        elif domain == "linear_algebra":
            # For systems of linear equations with 2 or 3 variables, show geometric representation
            # This requires specialized handling based on the number of variables
            free_vars = eq_expr.free_symbols
            if len(free_vars) == 2:
                request = {
                    "visualization_type": "linear_system_2d",
                    "equations": [sympy_to_latex(eq_expr)],
                    "solutions": [sympy_to_latex(sol) for sol in sol_exprs],
                    "title": "System of Linear Equations",
                    "show_grid": True
                }
            elif len(free_vars) == 3:
                request = {
                    "visualization_type": "linear_system_3d",
                    "equations": [sympy_to_latex(eq_expr)],
                    "solutions": [sympy_to_latex(sol) for sol in sol_exprs],
                    "title": "System of Linear Equations (3D)",
                    "show_grid": True
                }
            else:
                # Default to text representation for higher dimensions
                return {
                    "visualization_type": "text",
                    "message": "Visualization not available for systems with more than 3 variables"
                }
        else:
            # Default visualization based on the number of variables
            return self.visualize_expression(
                eq_expr, 
                title="Equation Visualization",
                solutions=solutions
            )
            
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def _visualize_series(self, 
                         expression: Union[str, sp.Expr],
                         series_result: Union[str, sp.Expr],
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for a series expansion.
        
        Args:
            expression: Original expression
            series_result: Series expansion result
            context: Additional context
            
        Returns:
            Visualization result
        """
        expr = self._ensure_sympy_expression(expression)
        series = self._ensure_sympy_expression(series_result)
        
        # Get the order of the series if available
        order = context.get("order", 5) if context else 5
        
        # Create visualization request
        request = {
            "visualization_type": "series_approximation",
            "original_function": sympy_to_latex(expr),
            "series_expansion": sympy_to_latex(series),
            "order": order,
            "title": f"Function and its Series Expansion (Order {order})",
            "show_grid": True,
            "x_range": [-2, 2]  # Typically series are centered around x=0
        }
        
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def _visualize_limit(self, 
                        expression: Union[str, sp.Expr],
                        limit_result: Union[str, sp.Expr, float],
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for a limit computation.
        
        Args:
            expression: Original expression
            limit_result: Limit result
            context: Additional context
            
        Returns:
            Visualization result
        """
        expr = self._ensure_sympy_expression(expression)
        
        # Extract limit point from context
        limit_point = context.get("limit_point", 0) if context else 0
        
        # Create visualization request
        request = {
            "visualization_type": "limit_visualization",
            "expression": sympy_to_latex(expr),
            "limit_point": float(limit_point),
            "limit_value": float(limit_result) if isinstance(limit_result, (int, float)) else str(limit_result),
            "title": f"Limit of Function as x approaches {limit_point}",
            "show_grid": True,
            "highlight_point": True
        }
        
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def _visualize_by_domain(self, 
                            expression: Union[str, sp.Expr],
                            result: Union[str, sp.Expr],
                            domain: str,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create domain-specific visualization.
        
        Args:
            expression: Original expression
            result: Computation result
            domain: Mathematical domain
            context: Additional context
            
        Returns:
            Visualization result
        """
        if domain == "statistics":
            return self._visualize_statistical(expression, result, context)
        elif domain == "linear_algebra":
            return self._visualize_linear_algebra(expression, result, context)
        elif domain == "calculus":
            return self._visualize_calculus(expression, result, context)
        else:
            # Default to general visualization
            return self.visualize_expression(
                expression,
                title=f"{domain.capitalize()} Visualization" if domain else "Mathematical Visualization"
            )
    
    def _visualize_statistical(self, 
                              expression: Union[str, sp.Expr],
                              result: Union[str, sp.Expr],
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for statistical computations.
        
        Args:
            expression: Original expression
            result: Computation result
            context: Additional context
            
        Returns:
            Visualization result
        """
        # For statistical visualizations, we need to know what type of data/distribution we're dealing with
        data_type = context.get("data_type") if context else None
        
        if data_type == "distribution":
            # Visualize a probability distribution
            distribution = context.get("distribution")
            params = context.get("parameters", {})
            
            request = {
                "visualization_type": "probability_distribution",
                "distribution": distribution,
                "parameters": params,
                "title": f"{distribution.capitalize()} Distribution",
                "show_grid": True
            }
        elif data_type == "dataset":
            # Visualize dataset statistics
            dataset = context.get("dataset")
            
            # Determine what type of chart to use based on the data
            if len(dataset) > 0 and all(isinstance(x, (int, float)) for x in dataset):
                request = {
                    "visualization_type": "statistical",
                    "chart_type": "histogram",
                    "data": dataset,
                    "title": "Data Distribution",
                    "x_label": "Value",
                    "y_label": "Frequency"
                }
            else:
                return {
                    "visualization_type": "text",
                    "message": "Cannot visualize non-numerical dataset"
                }
        else:
            # Default statistical visualization
            return {
                "visualization_type": "text",
                "message": "Insufficient information for statistical visualization"
            }
            
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def _visualize_linear_algebra(self, 
                                 expression: Union[str, sp.Expr],
                                 result: Union[str, sp.Expr],
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for linear algebra computations.
        
        Args:
            expression: Original expression
            result: Computation result
            context: Additional context
            
        Returns:
            Visualization result
        """
        # For linear algebra, determine what type of object we're dealing with
        object_type = context.get("object_type") if context else None
        
        if object_type == "matrix":
            # Visualize a matrix
            matrix = context.get("matrix")
            
            request = {
                "visualization_type": "matrix_visualization",
                "matrix": matrix,
                "title": "Matrix Visualization",
            }
            
            # Add eigenvalues/eigenvectors if available
            if "eigenvalues" in context:
                request["eigenvalues"] = context["eigenvalues"]
            if "eigenvectors" in context:
                request["eigenvectors"] = context["eigenvectors"]
                
        elif object_type == "vector":
            # Visualize vectors
            vectors = context.get("vectors", [])
            
            # Determine dimensionality
            if all(len(v) == 2 for v in vectors):
                request = {
                    "visualization_type": "vector_2d",
                    "vectors": vectors,
                    "title": "Vector Visualization",
                    "show_grid": True
                }
            elif all(len(v) == 3 for v in vectors):
                request = {
                    "visualization_type": "vector_3d",
                    "vectors": vectors,
                    "title": "Vector Visualization (3D)",
                    "show_grid": True
                }
            else:
                return {
                    "visualization_type": "text",
                    "message": "Cannot visualize vectors of dimension other than 2 or 3"
                }
                
        elif object_type == "transformation":
            # Visualize a linear transformation
            matrix = context.get("matrix")
            
            if len(matrix) == 2 and len(matrix[0]) == 2:
                request = {
                    "visualization_type": "linear_transformation_2d",
                    "matrix": matrix,
                    "title": "Linear Transformation",
                    "show_grid": True,
                    "show_unit_vectors": True
                }
            elif len(matrix) == 3 and len(matrix[0]) == 3:
                request = {
                    "visualization_type": "linear_transformation_3d",
                    "matrix": matrix,
                    "title": "Linear Transformation (3D)",
                    "show_grid": True
                }
            else:
                return {
                    "visualization_type": "text",
                    "message": "Cannot visualize transformations of dimension other than 2 or 3"
                }
        else:
            # Default linear algebra visualization
            return {
                "visualization_type": "text",
                "message": "Insufficient information for linear algebra visualization"
            }
            
        # Send request to visualization agent
        response = self.message_bus.send_request_sync(
            recipient="visualization_agent",
            message_body=request,
            message_type="visualization_request"
        )
        
        return response
    
    def _visualize_calculus(self, 
                           expression: Union[str, sp.Expr],
                           result: Union[str, sp.Expr],
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create visualization for general calculus computations.
        
        Args:
            expression: Original expression
            result: Computation result
            context: Additional context
            
        Returns:
            Visualization result
        """
        # Default to a general function visualization
        return self.visualize_expression(
            expression,
            title="Calculus Function Visualization"
        )
    
    def _identify_key_steps(self, solution_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify key steps in a solution that would benefit from visualization.
        
        Args:
            solution_steps: List of solution steps
            
        Returns:
            List of steps that should be visualized
        """
        key_steps = []
        
        # Select steps based on heuristics
        for i, step in enumerate(solution_steps):
            operation = step.get("operation")
            
            # Visualization-worthy operations
            if operation in ["solve", "differentiate", "integrate", "substitute", "simplify"]:
                # Only visualize significant changes or final results
                if i == len(solution_steps) - 1:  # Final step
                    key_steps.append(step)
                elif i == 0:  # First step
                    key_steps.append(step)
                elif i > 0 and i < len(solution_steps) - 1:
                    # Middle steps - only include significant changes
                    curr_expr = step.get("output")
                    prev_expr = solution_steps[i-1].get("output")
                    if curr_expr and prev_expr and curr_expr != prev_expr:
                        # Check if this is a significant change
                        # For now, include every other step to avoid too many visualizations
                        if len(key_steps) == 0 or i - key_steps[-1].get("step_number", 0) >= 2:
                            key_steps.append(step)
        
        # Limit the number of visualizations to avoid overwhelming the user
        if len(key_steps) > 5:
            # Keep first, last, and a few key middle steps
            first_step = key_steps[0]
            last_step = key_steps[-1]
            
            # Select a few evenly spaced middle steps
            step_indices = [int(len(key_steps) * i / 4) for i in range(1, 4)]
            middle_steps = [key_steps[i] for i in step_indices if 0 < i < len(key_steps) - 1]
            
            key_steps = [first_step] + middle_steps + [last_step]
            
        return key_steps
from typing import Dict, Any, List, Optional, Union, Tuple
import sympy as sp

class VisualizationSelector:
    """
    Analyzes mathematical context to determine appropriate visualizations.
    """
    
    def __init__(self):
        """Initialize the visualization selector."""
        # Define visualization types and their requirements
        self.visualization_types = {
            "function_2d": self._is_suitable_for_2d_function,
            "function_3d": self._is_suitable_for_3d_function,
            "derivative": self._is_suitable_for_derivative,
            "critical_points": self._is_suitable_for_critical_points,
            "integral": self._is_suitable_for_integral,
            "vector_field": self._is_suitable_for_vector_field,
            "histogram": self._is_suitable_for_histogram,
            "scatter": self._is_suitable_for_scatter
        }
    
    def select_visualization(self, mathematical_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate visualization based on mathematical context.
        
        Args:
            mathematical_context: Dictionary containing mathematical information
            
        Returns:
            Dictionary with recommended visualization type and parameters
        """
        # Check each visualization type for suitability
        suitable_visualizations = []
        for viz_type, check_function in self.visualization_types.items():
            result = check_function(mathematical_context)
            if result["suitable"]:
                suitable_visualizations.append({
                    "type": viz_type,
                    "params": result["params"],
                    "suitability_score": result["suitability_score"]
                })
        
        # Sort by suitability score (higher is better)
        suitable_visualizations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        if suitable_visualizations:
            return {
                "recommended_visualization": suitable_visualizations[0],
                "alternative_visualizations": suitable_visualizations[1:],
                "success": True
            }
        else:
            return {
                "success": False,
                "error": "No suitable visualization found for the given context",
                "fallback": {
                    "type": "text",
                    "params": {
                        "message": "No suitable visualization available for this expression."
                    }
                }
            }
    
    def _detect_variables(self, expression: Union[str, sp.Expr]) -> List[str]:
        """
        Detect variables in a mathematical expression.
        
        Args:
            expression: Mathematical expression
            
        Returns:
            List of variable names
        """
        if isinstance(expression, str):
            try:
                # Convert string to SymPy expression
                expr = sp.sympify(expression)
            except Exception:
                # Return empty list if parsing fails
                return []
        else:
            expr = expression
        
        # Extract free symbols
        try:
            symbols = list(expr.free_symbols)
            return [str(symbol) for symbol in symbols]
        except Exception:
            return []
    
    def _is_suitable_for_2d_function(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for 2D function plot."""
        # Check if we have an expression
        expression = context.get("expression")
        if not expression:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Detect variables
        variables = self._detect_variables(expression)
        
        # Ideal for single-variable expressions
        if len(variables) == 1:
            return {
                "suitable": True,
                "suitability_score": 0.9,
                "params": {
                    "expression": expression,
                    "x_range": context.get("x_range", (-10, 10)),
                    "variable": variables[0]
                }
            }
        # Could work for multi-variable expressions with substitutions
        elif len(variables) > 1:
            # Check if the context specifies variable assignments
            assignments = context.get("variable_assignments", {})
            remaining_vars = [v for v in variables if v not in assignments]
            
            if len(remaining_vars) == 1:
                return {
                    "suitable": True,
                    "suitability_score": 0.7,
                    "params": {
                        "expression": expression,
                        "x_range": context.get("x_range", (-10, 10)),
                        "variable": remaining_vars[0],
                        "substitutions": assignments
                    }
                }
        
        return {"suitable": False, "suitability_score": 0, "params": {}}
    
    def _is_suitable_for_3d_function(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for 3D function plot."""
        # Check if we have an expression
        expression = context.get("expression")
        if not expression:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Detect variables
        variables = self._detect_variables(expression)
        
        # Ideal for two-variable expressions
        if len(variables) == 2:
            return {
                "suitable": True,
                "suitability_score": 0.9,
                "params": {
                    "expression": expression,
                    "x_range": context.get("x_range", (-5, 5)),
                    "y_range": context.get("y_range", (-5, 5))
                }
            }
        # Could work for multi-variable expressions with substitutions
        elif len(variables) > 2:
            # Check if the context specifies variable assignments
            assignments = context.get("variable_assignments", {})
            remaining_vars = [v for v in variables if v not in assignments]
            
            if len(remaining_vars) == 2:
                return {
                    "suitable": True,
                    "suitability_score": 0.7,
                    "params": {
                        "expression": expression,
                        "x_range": context.get("x_range", (-5, 5)),
                        "y_range": context.get("y_range", (-5, 5)),
                        "substitutions": assignments
                    }
                }
        
        return {"suitable": False, "suitability_score": 0, "params": {}}
    
    def _is_suitable_for_derivative(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for derivative visualization."""
        # Check if we have an expression and operation indicates differentiation
        expression = context.get("expression")
        operation = context.get("operation", "").lower()
        domain = context.get("domain", "").lower()
        
        derivative_indicators = ["derivative", "differentiate", "diff", "find_derivative"]
        
        if not expression:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check if operation or domain indicate a derivative
        is_derivative_op = any(op in operation for op in derivative_indicators)
        is_calculus_domain = "calculus" in domain
        
        # Detect variables
        variables = self._detect_variables(expression)
        
        # Suitable for single-variable expressions with derivative operation
        if len(variables) == 1 and (is_derivative_op or is_calculus_domain):
            return {
                "suitable": True,
                "suitability_score": 0.8 if is_derivative_op else 0.6,
                "params": {
                    "expression": expression,
                    "variable": variables[0],
                    "x_range": context.get("x_range", (-10, 10))
                }
            }
        
        return {"suitable": False, "suitability_score": 0, "params": {}}
    
    def _is_suitable_for_critical_points(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for critical points visualization."""
        # Check if we have an expression
        expression = context.get("expression")
        operation = context.get("operation", "").lower()
        query = context.get("query", "").lower()
        
        critical_point_indicators = ["critical", "extrema", "maximum", "minimum", "inflection"]
        
        if not expression:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check if operation or query indicate finding critical points
        is_critical_op = any(indicator in operation for indicator in critical_point_indicators)
        is_critical_query = any(indicator in query for indicator in critical_point_indicators)
        
        # Detect variables
        variables = self._detect_variables(expression)
        
        # Suitable for single-variable expressions with critical point indicators
        if len(variables) == 1 and (is_critical_op or is_critical_query):
            return {
                "suitable": True,
                "suitability_score": 0.85 if is_critical_op else 0.7,
                "params": {
                    "expression": expression,
                    "variable": variables[0],
                    "x_range": context.get("x_range", (-10, 10))
                }
            }
        
        return {"suitable": False, "suitability_score": 0, "params": {}}
    
    def _is_suitable_for_integral(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for integral visualization."""
        # Check if we have an expression and operation indicates integration
        expression = context.get("expression")
        operation = context.get("operation", "").lower()
        domain = context.get("domain", "").lower()
        query = context.get("query", "").lower()
        
        integral_indicators = ["integral", "integrate", "antiderivative"]
        
        if not expression:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check for bounds in the context
        lower_bound = context.get("lower_bound")
        upper_bound = context.get("upper_bound")
        
        # Check if operation or domain indicate an integral
        is_integral_op = any(op in operation for op in integral_indicators)
        is_calculus_domain = "calculus" in domain
        is_integral_query = any(op in query for op in integral_indicators)
        
        # Detect variables
        variables = self._detect_variables(expression)
        
        # Suitable for single-variable expressions with integration operation and bounds
        if len(variables) == 1 and (is_integral_op or is_calculus_domain or is_integral_query):
            # Definite integral visualization requires bounds
            if lower_bound is not None and upper_bound is not None:
                return {
                    "suitable": True,
                    "suitability_score": 0.9 if is_integral_op else 0.7,
                    "params": {
                        "expression": expression,
                        "variable": variables[0],
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound
                    }
                }
            # Without bounds, suggest a function visualization instead
            else:
                return {
                    "suitable": True,
                    "suitability_score": 0.6,
                    "params": {
                        "expression": expression,
                        "variable": variables[0],
                        "x_range": context.get("x_range", (-10, 10)),
                        "message": "Visualization shows the function without integration bounds"
                    }
                }
        
        return {"suitable": False, "suitability_score": 0, "params": {}}
    
    def _is_suitable_for_vector_field(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for vector field visualization."""
        # Check if we have vector components or a vector field
        x_component = context.get("x_component")
        y_component = context.get("y_component")
        vector_field = context.get("vector_field")
        operation = context.get("operation", "").lower()
        domain = context.get("domain", "").lower()
        
        vector_field_indicators = ["vector field", "gradient", "curl", "divergence", "flow"]
        
        # Try to extract components from vector field if provided
        if not (x_component and y_component) and vector_field:
            try:
                components = vector_field.split(",")
                if len(components) == 2:
                    x_component = components[0].strip()
                    y_component = components[1].strip()
            except:
                pass
        
        if not (x_component and y_component):
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check if domain or operation indicates vector field
        is_vector_field_op = any(indicator in operation for indicator in vector_field_indicators)
        is_vector_field_domain = any(indicator in domain for indicator in ["vector", "fluid", "electromagnetic"])
        
        # Suitable for vector field with components
        suitability_score = 0.7
        if is_vector_field_op:
            suitability_score = 0.9
        elif is_vector_field_domain:
            suitability_score = 0.8
        
        return {
            "suitable": True,
            "suitability_score": suitability_score,
            "params": {
                "x_component": x_component,
                "y_component": y_component,
                "x_range": context.get("x_range", (-5, 5)),
                "y_range": context.get("y_range", (-5, 5)),
                "grid_density": context.get("grid_density", 20)
            }
        }
    
    def _is_suitable_for_histogram(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for histogram visualization."""
        # Check if we have numerical data
        data = context.get("data")
        data_type = context.get("data_type", "").lower()
        domain = context.get("domain", "").lower()
        query = context.get("query", "").lower()
        
        histogram_indicators = ["histogram", "distribution", "frequency", "count"]
        statistics_indicators = ["statistics", "statistical", "stats", "probability"]
        
        if data is None:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check if data is a list or can be treated as numerical data
        is_numerical_data = False
        try:
            if isinstance(data, list) and len(data) > 0:
                # Check if at least 80% of items can be converted to float
                valid_count = sum(1 for item in data if isinstance(item, (int, float)) or 
                                 (isinstance(item, str) and item.replace('.', '', 1).isdigit()))
                is_numerical_data = valid_count / len(data) >= 0.8
        except:
            pass
        
        if not is_numerical_data:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check if domain, data_type, or query indicate histogram
        is_histogram_type = any(indicator in data_type for indicator in histogram_indicators)
        is_statistics_domain = any(indicator in domain for indicator in statistics_indicators)
        is_histogram_query = any(indicator in query for indicator in histogram_indicators)
        
        # Suitable for numerical data with histogram indicators
        suitability_score = 0.7
        if is_histogram_type:
            suitability_score = 0.9
        elif is_statistics_domain or is_histogram_query:
            suitability_score = 0.8
        
        return {
            "suitable": True,
            "suitability_score": suitability_score,
            "params": {
                "data": data,
                "bins": context.get("bins", "auto"),
                "title": context.get("title"),
                "x_label": context.get("x_label", "Value"),
                "y_label": context.get("y_label", "Frequency"),
                "show_kde": context.get("show_kde", False)
            }
        }
    
    def _is_suitable_for_scatter(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if context is suitable for scatter plot visualization."""
        # Check if we have paired data
        x_data = context.get("x_data")
        y_data = context.get("y_data")
        data = context.get("data")
        data_type = context.get("data_type", "").lower()
        query = context.get("query", "").lower()
        
        scatter_indicators = ["scatter", "correlation", "relationship", "regression"]
        
        # Try to extract x and y data from data if provided
        if not (x_data and y_data) and data:
            try:
                if isinstance(data, list) and len(data) > 0:
                    # Check if data is list of pairs
                    if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in data):
                        x_data = [item[0] for item in data]
                        y_data = [item[1] for item in data]
            except:
                pass
        
        if not (x_data and y_data):
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check if data lengths match
        try:
            if len(x_data) != len(y_data):
                return {"suitable": False, "suitability_score": 0, "params": {}}
        except:
            return {"suitable": False, "suitability_score": 0, "params": {}}
        
        # Check if data_type or query indicate scatter plot
        is_scatter_type = any(indicator in data_type for indicator in scatter_indicators)
        is_scatter_query = any(indicator in query for indicator in scatter_indicators)
        
        # Suitable for paired data with scatter indicators
        suitability_score = 0.7
        if is_scatter_type:
            suitability_score = 0.9
        elif is_scatter_query:
            suitability_score = 0.8
        
        return {
            "suitable": True,
            "suitability_score": suitability_score,
            "params": {
                "x_data": x_data,
                "y_data": y_data,
                "title": context.get("title"),
                "x_label": context.get("x_label", "X"),
                "y_label": context.get("y_label", "Y"),
                "show_regression": context.get("show_regression", True)
            }
        }

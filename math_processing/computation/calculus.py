"""
Calculus Module - specialized calculus operations using SymPy.

This module implements calculus operations including differentiation, integration,
limits, and series expansions.
"""

import sympy as sp
from typing import Dict, List, Union, Any, Optional, Tuple
import logging


class CalculusProcessor:
    """Processor for calculus operations."""
    
    def __init__(self):
        """Initialize the calculus processor."""
        self.logger = logging.getLogger(__name__)
    
    def differentiate(self, 
                     expression: Union[sp.Expr, str], 
                     variable: Union[sp.Symbol, str], 
                     order: int = 1,
                     steps: bool = True) -> Dict[str, Any]:
        """
        Differentiate an expression with respect to a variable.
        
        Args:
            expression: Expression to differentiate
            variable: Variable to differentiate with respect to
            order: Order of differentiation
            steps: Whether to generate steps
            
        Returns:
            Dictionary with differentiation information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Compute the derivative
            derivative = sp.diff(expression, variable, order)
            
            # Generate steps if requested
            if steps:
                diff_steps = self._generate_differentiation_steps(expression, variable, order, derivative)
            else:
                diff_steps = []
            
            return {
                "success": True,
                "derivative": derivative,
                "order": order,
                "variable": variable,
                "steps": diff_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in differentiation: {str(e)}")
            return {
                "success": False,
                "derivative": None,
                "steps": None,
                "error": str(e)
            }
    
    def integrate(self, 
                 expression: Union[sp.Expr, str], 
                 variable: Union[sp.Symbol, str], 
                 lower_bound: Optional[Union[sp.Expr, str, float]] = None, 
                 upper_bound: Optional[Union[sp.Expr, str, float]] = None,
                 steps: bool = True) -> Dict[str, Any]:
        """
        Integrate an expression with respect to a variable.
        
        Args:
            expression: Expression to integrate
            variable: Variable to integrate with respect to
            lower_bound: Lower bound for definite integration (optional)
            upper_bound: Upper bound for definite integration (optional)
            steps: Whether to generate steps
            
        Returns:
            Dictionary with integration information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Handle string bounds
            if isinstance(lower_bound, str):
                lower_bound = sp.sympify(lower_bound)
            if isinstance(upper_bound, str):
                upper_bound = sp.sympify(upper_bound)
            
            # Determine if this is a definite or indefinite integral
            if lower_bound is not None and upper_bound is not None:
                # Definite integral
                integral = sp.integrate(expression, (variable, lower_bound, upper_bound))
                definite = True
            else:
                # Indefinite integral
                integral = sp.integrate(expression, variable)
                definite = False
            
            # Generate steps if requested
            if steps:
                int_steps = self._generate_integration_steps(expression, variable, integral, lower_bound, upper_bound)
            else:
                int_steps = []
            
            return {
                "success": True,
                "integral": integral,
                "definite": definite,
                "variable": variable,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "steps": int_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in integration: {str(e)}")
            return {
                "success": False,
                "integral": None,
                "steps": None,
                "error": str(e)
            }
    
    def compute_limit(self, 
                     expression: Union[sp.Expr, str], 
                     variable: Union[sp.Symbol, str], 
                     point: Union[sp.Expr, str, float],
                     direction: str = "both",
                     steps: bool = True) -> Dict[str, Any]:
        """
        Compute the limit of an expression as a variable approaches a point.
        
        Args:
            expression: Expression to find the limit of
            variable: Variable for the limit
            point: Point that the variable approaches
            direction: Direction of approach ("+" for right, "-" for left, "both" for both)
            steps: Whether to generate steps
            
        Returns:
            Dictionary with limit information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Handle string point
            if isinstance(point, str):
                point = sp.sympify(point)
            
            # Compute the limit with specified direction
            if direction == "+":
                limit = sp.limit(expression, variable, point, dir="+")
                dir_symbol = "^+"
            elif direction == "-":
                limit = sp.limit(expression, variable, point, dir="-")
                dir_symbol = "^-"
            else:  # direction == "both" or any other value
                limit = sp.limit(expression, variable, point)
                dir_symbol = ""
            
            # Generate steps if requested
            if steps:
                limit_steps = self._generate_limit_steps(expression, variable, point, limit, dir_symbol)
            else:
                limit_steps = []
            
            return {
                "success": True,
                "limit": limit,
                "variable": variable,
                "point": point,
                "direction": direction,
                "steps": limit_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in limit computation: {str(e)}")
            return {
                "success": False,
                "limit": None,
                "steps": None,
                "error": str(e)
            }
    
    def series_expansion(self, 
                        expression: Union[sp.Expr, str], 
                        variable: Union[sp.Symbol, str], 
                        point: Union[sp.Expr, str, float],
                        order: int = 5,
                        steps: bool = True) -> Dict[str, Any]:
        """
        Compute the Taylor/Maclaurin series expansion of an expression.
        
        Args:
            expression: Expression to expand in series
            variable: Variable for the expansion
            point: Point around which to expand
            order: Order of the series expansion
            steps: Whether to generate steps
            
        Returns:
            Dictionary with series expansion information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Handle string point
            if isinstance(point, str):
                point = sp.sympify(point)
            
            # Compute the series expansion
            series = expression.series(variable, point, order).removeO()
            
            # Generate steps if requested
            if steps:
                series_steps = self._generate_series_steps(expression, variable, point, order, series)
            else:
                series_steps = []
            
            return {
                "success": True,
                "series": series,
                "variable": variable,
                "point": point,
                "order": order,
                "steps": series_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in series expansion: {str(e)}")
            return {
                "success": False,
                "series": None,
                "steps": None,
                "error": str(e)
            }
    
    def _generate_differentiation_steps(self, 
                                      expression: sp.Expr, 
                                      variable: sp.Symbol, 
                                      order: int, 
                                      derivative: sp.Expr) -> List[Dict[str, str]]:
        """
        Generate steps for differentiation.
        
        Args:
            expression: Original expression
            variable: Variable of differentiation
            order: Order of differentiation
            derivative: Computed derivative
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        steps.append({
            "explanation": "Start with the original expression",
            "expression": sp.latex(expression)
        })
        
        # For higher-order derivatives, compute each step
        current_expr = expression
        for i in range(1, order + 1):
            current_derivative = sp.diff(current_expr, variable)
            
            # Add a step for this derivative
            if order > 1:
                steps.append({
                    "explanation": f"Compute the derivative of order {i}",
                    "expression": f"\\frac{{d^{i}}}{{d{sp.latex(variable)}^{i}}}[{sp.latex(expression)}] = {sp.latex(current_derivative)}"
                })
            else:
                steps.append({
                    "explanation": "Apply the rules of differentiation",
                    "expression": f"\\frac{{d}}{{d{sp.latex(variable)}}}[{sp.latex(expression)}] = {sp.latex(current_derivative)}"
                })
            
            current_expr = current_derivative
        
        return steps
    
    def _generate_integration_steps(self, 
                                  expression: sp.Expr, 
                                  variable: sp.Symbol, 
                                  integral: sp.Expr, 
                                  lower_bound: Optional[sp.Expr] = None, 
                                  upper_bound: Optional[sp.Expr] = None) -> List[Dict[str, str]]:
        """
        Generate steps for integration.
        
        Args:
            expression: Original expression
            variable: Variable of integration
            integral: Computed integral
            lower_bound: Lower bound for definite integration (optional)
            upper_bound: Upper bound for definite integration (optional)
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        if lower_bound is not None and upper_bound is not None:
            steps.append({
                "explanation": "Start with the definite integral",
                "expression": f"\\int_{{{sp.latex(lower_bound)}}}^{{{sp.latex(upper_bound)}}} {sp.latex(expression)} \\, d{sp.latex(variable)}"
            })
        else:
            steps.append({
                "explanation": "Start with the indefinite integral",
                "expression": f"\\int {sp.latex(expression)} \\, d{sp.latex(variable)}"
            })
        
        # Step 2: Apply integration rules
        # In a real implementation, we would detect what rules are being applied
        # and provide more specific steps. This is a simplified version.
        
        # Check if it's a basic function
        if expression.is_polynomial(variable):
            steps.append({
                "explanation": "Apply the power rule for integration",
                "expression": f"\\int {sp.latex(variable)}^n \\, d{sp.latex(variable)} = \\frac{{{sp.latex(variable)}^{{n+1}}}}{{n+1}} + C"
            })
        elif expression.has(sp.sin(variable)) or expression.has(sp.cos(variable)):
            steps.append({
                "explanation": "Apply trigonometric integration rules",
                "expression": "\\text{Use standard trigonometric integration formulas}"
            })
        elif expression.has(sp.exp(variable)):
            steps.append({
                "explanation": "Apply the exponential integration rule",
                "expression": f"\\int e^{{{sp.latex(variable)}}} \\, d{sp.latex(variable)} = e^{{{sp.latex(variable)}}} + C"
            })
        else:
            steps.append({
                "explanation": "Apply appropriate integration techniques",
                "expression": "\\text{(Integration steps may involve substitution, parts, or other methods)}"
            })
        
        # Step 3: Result of indefinite integration
        if lower_bound is None or upper_bound is None:
            steps.append({
                "explanation": "The indefinite integral is",
                "expression": f"{sp.latex(integral)} + C"
            })
        else:
            # Step 3a: Show antiderivative for definite integral
            antiderivative = sp.integrate(expression, variable)
            steps.append({
                "explanation": "Find the antiderivative",
                "expression": f"F({sp.latex(variable)}) = {sp.latex(antiderivative)}"
            })
            
            # Step 3b: Evaluate at the bounds
            upper_value = antiderivative.subs(variable, upper_bound)
            lower_value = antiderivative.subs(variable, lower_bound)
            
            steps.append({
                "explanation": "Evaluate the antiderivative at the upper and lower bounds",
                "expression": f"F({sp.latex(upper_bound)}) - F({sp.latex(lower_bound)}) = {sp.latex(upper_value)} - {sp.latex(lower_value)} = {sp.latex(integral)}"
            })
        
        return steps
    
    def _generate_limit_steps(self, 
                            expression: sp.Expr, 
                            variable: sp.Symbol, 
                            point: sp.Expr, 
                            limit: sp.Expr, 
                            dir_symbol: str) -> List[Dict[str, str]]:
        """
        Generate steps for limit computation.
        
        Args:
            expression: Original expression
            variable: Limit variable
            point: Point of approach
            limit: Computed limit
            dir_symbol: Direction symbol ("^+", "^-", or "")
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original limit expression
        steps.append({
            "explanation": "Start with the limit expression",
            "expression": f"\\lim_{{{sp.latex(variable)} \\to {sp.latex(point)}{dir_symbol}}} {sp.latex(expression)}"
        })
        
        # Step 2: Check for simple substitution
        try:
            direct_sub = expression.subs(variable, point)
            if not direct_sub.has(sp.nan) and not direct_sub.has(sp.oo) and not direct_sub.has(-sp.oo):
                steps.append({
                    "explanation": "Try direct substitution of the limit value",
                    "expression": f"{sp.latex(expression)}|_{{{sp.latex(variable)}={sp.latex(point)}}} = {sp.latex(direct_sub)}"
                })
                
                if direct_sub == limit:
                    steps.append({
                        "explanation": "Direct substitution yields the limit",
                        "expression": f"\\lim_{{{sp.latex(variable)} \\to {sp.latex(point)}{dir_symbol}}} {sp.latex(expression)} = {sp.latex(limit)}"
                    })
                    return steps
            else:
                steps.append({
                    "explanation": "Direct substitution leads to an indeterminate form",
                    "expression": f"{sp.latex(expression)}|_{{{sp.latex(variable)}={sp.latex(point)}}} = \\text{{indeterminate}}"
                })
        except:
            steps.append({
                "explanation": "Direct substitution is not applicable",
                "expression": "\\text{(Need to use limit techniques)}"
            })
        
        # Step 3: Apply techniques based on expression type
        # This is a simplified implementation
        if expression.is_rational_function(variable):
            steps.append({
                "explanation": "For rational functions, factor the numerator and denominator",
                "expression": "\\text{(Factorization steps omitted)}"
            })
        elif expression.has(sp.sin(variable)) or expression.has(sp.cos(variable)):
            steps.append({
                "explanation": "For trigonometric limits, special limit formulas may apply",
                "expression": "\\text{e.g., } \\lim_{x \\to 0} \\frac{\\sin x}{x} = 1"
            })
        else:
            steps.append({
                "explanation": "Apply appropriate limit techniques",
                "expression": "\\text{(L'Hôpital's rule, algebraic manipulation, etc.)}"
            })
        
        # Step 4: Final limit value
        steps.append({
            "explanation": "The limit evaluates to",
            "expression": f"\\lim_{{{sp.latex(variable)} \\to {sp.latex(point)}{dir_symbol}}} {sp.latex(expression)} = {sp.latex(limit)}"
        })
        
        return steps
    
    def _generate_series_steps(self, 
                             expression: sp.Expr, 
                             variable: sp.Symbol, 
                             point: sp.Expr, 
                             order: int, 
                             series: sp.Expr) -> List[Dict[str, str]]:
        """
        Generate steps for series expansion.
        
        Args:
            expression: Original expression
            variable: Series variable
            point: Expansion point
            order: Expansion order
            series: Computed series
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        if point == 0:
            steps.append({
                "explanation": f"Find the Maclaurin series expansion of order {order}",
                "expression": f"{sp.latex(expression)}"
            })
        else:
            steps.append({
                "explanation": f"Find the Taylor series expansion of order {order} around {sp.latex(point)}",
                "expression": f"{sp.latex(expression)}"
            })
        
        # Step 2: Taylor series formula
        steps.append({
            "explanation": "The Taylor series formula is",
            "expression": f"f({sp.latex(variable)}) = \\sum_{{n=0}}^{{\\infty}} \\frac{{f^{{(n)}}({sp.latex(point)})}}{{n!}}({sp.latex(variable)}-{sp.latex(point)})^n"
        })
        
        # Step 3: Compute derivatives at the point
        # In a real implementation, we would show the derivatives
        # This is a simplified version
        steps.append({
            "explanation": "Compute the derivatives at the expansion point",
            "expression": "f({sp.latex(point)}), f'({sp.latex(point)}), f''({sp.latex(point)}), \\ldots"
        })
        
        # Step 4: Final series expansion
        steps.append({
            "explanation": f"The series expansion up to order {order} is",
            "expression": f"{sp.latex(expression)} = {sp.latex(series)} + O(({sp.latex(variable)}-{sp.latex(point)})^{{{order+1}}})"
        })
        
        return steps
EOFcat > math_processing/computation/calculus.py << 'EOF'
"""
Calculus Module - specialized calculus operations using SymPy.

This module implements calculus operations including differentiation, integration,
limits, and series expansions.
"""

import sympy as sp
from typing import Dict, List, Union, Any, Optional, Tuple
import logging


class CalculusProcessor:
    """Processor for calculus operations."""
    
    def __init__(self):
        """Initialize the calculus processor."""
        self.logger = logging.getLogger(__name__)
    
    def differentiate(self, 
                     expression: Union[sp.Expr, str], 
                     variable: Union[sp.Symbol, str], 
                     order: int = 1,
                     steps: bool = True) -> Dict[str, Any]:
        """
        Differentiate an expression with respect to a variable.
        
        Args:
            expression: Expression to differentiate
            variable: Variable to differentiate with respect to
            order: Order of differentiation
            steps: Whether to generate steps
            
        Returns:
            Dictionary with differentiation information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Compute the derivative
            derivative = sp.diff(expression, variable, order)
            
            # Generate steps if requested
            if steps:
                diff_steps = self._generate_differentiation_steps(expression, variable, order, derivative)
            else:
                diff_steps = []
            
            return {
                "success": True,
                "derivative": derivative,
                "order": order,
                "variable": variable,
                "steps": diff_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in differentiation: {str(e)}")
            return {
                "success": False,
                "derivative": None,
                "steps": None,
                "error": str(e)
            }
    
    def integrate(self, 
                 expression: Union[sp.Expr, str], 
                 variable: Union[sp.Symbol, str], 
                 lower_bound: Optional[Union[sp.Expr, str, float]] = None, 
                 upper_bound: Optional[Union[sp.Expr, str, float]] = None,
                 steps: bool = True) -> Dict[str, Any]:
        """
        Integrate an expression with respect to a variable.
        
        Args:
            expression: Expression to integrate
            variable: Variable to integrate with respect to
            lower_bound: Lower bound for definite integration (optional)
            upper_bound: Upper bound for definite integration (optional)
            steps: Whether to generate steps
            
        Returns:
            Dictionary with integration information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Handle string bounds
            if isinstance(lower_bound, str):
                lower_bound = sp.sympify(lower_bound)
            if isinstance(upper_bound, str):
                upper_bound = sp.sympify(upper_bound)
            
            # Determine if this is a definite or indefinite integral
            if lower_bound is not None and upper_bound is not None:
                # Definite integral
                integral = sp.integrate(expression, (variable, lower_bound, upper_bound))
                definite = True
            else:
                # Indefinite integral
                integral = sp.integrate(expression, variable)
                definite = False
            
            # Generate steps if requested
            if steps:
                int_steps = self._generate_integration_steps(expression, variable, integral, lower_bound, upper_bound)
            else:
                int_steps = []
            
            return {
                "success": True,
                "integral": integral,
                "definite": definite,
                "variable": variable,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "steps": int_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in integration: {str(e)}")
            return {
                "success": False,
                "integral": None,
                "steps": None,
                "error": str(e)
            }
    
    def compute_limit(self, 
                     expression: Union[sp.Expr, str], 
                     variable: Union[sp.Symbol, str], 
                     point: Union[sp.Expr, str, float],
                     direction: str = "both",
                     steps: bool = True) -> Dict[str, Any]:
        """
        Compute the limit of an expression as a variable approaches a point.
        
        Args:
            expression: Expression to find the limit of
            variable: Variable for the limit
            point: Point that the variable approaches
            direction: Direction of approach ("+" for right, "-" for left, "both" for both)
            steps: Whether to generate steps
            
        Returns:
            Dictionary with limit information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Handle string point
            if isinstance(point, str):
                point = sp.sympify(point)
            
            # Compute the limit with specified direction
            if direction == "+":
                limit = sp.limit(expression, variable, point, dir="+")
                dir_symbol = "^+"
            elif direction == "-":
                limit = sp.limit(expression, variable, point, dir="-")
                dir_symbol = "^-"
            else:  # direction == "both" or any other value
                limit = sp.limit(expression, variable, point)
                dir_symbol = ""
            
            # Generate steps if requested
            if steps:
                limit_steps = self._generate_limit_steps(expression, variable, point, limit, dir_symbol)
            else:
                limit_steps = []
            
            return {
                "success": True,
                "limit": limit,
                "variable": variable,
                "point": point,
                "direction": direction,
                "steps": limit_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in limit computation: {str(e)}")
            return {
                "success": False,
                "limit": None,
                "steps": None,
                "error": str(e)
            }
    
    def series_expansion(self, 
                        expression: Union[sp.Expr, str], 
                        variable: Union[sp.Symbol, str], 
                        point: Union[sp.Expr, str, float],
                        order: int = 5,
                        steps: bool = True) -> Dict[str, Any]:
        """
        Compute the Taylor/Maclaurin series expansion of an expression.
        
        Args:
            expression: Expression to expand in series
            variable: Variable for the expansion
            point: Point around which to expand
            order: Order of the series expansion
            steps: Whether to generate steps
            
        Returns:
            Dictionary with series expansion information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Handle string point
            if isinstance(point, str):
                point = sp.sympify(point)
            
            # Compute the series expansion
            series = expression.series(variable, point, order).removeO()
            
            # Generate steps if requested
            if steps:
                series_steps = self._generate_series_steps(expression, variable, point, order, series)
            else:
                series_steps = []
            
            return {
                "success": True,
                "series": series,
                "variable": variable,
                "point": point,
                "order": order,
                "steps": series_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in series expansion: {str(e)}")
            return {
                "success": False,
                "series": None,
                "steps": None,
                "error": str(e)
            }
    
    def _generate_differentiation_steps(self, 
                                      expression: sp.Expr, 
                                      variable: sp.Symbol, 
                                      order: int, 
                                      derivative: sp.Expr) -> List[Dict[str, str]]:
        """
        Generate steps for differentiation.
        
        Args:
            expression: Original expression
            variable: Variable of differentiation
            order: Order of differentiation
            derivative: Computed derivative
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        steps.append({
            "explanation": "Start with the original expression",
            "expression": sp.latex(expression)
        })
        
        # For higher-order derivatives, compute each step
        current_expr = expression
        for i in range(1, order + 1):
            current_derivative = sp.diff(current_expr, variable)
            
            # Add a step for this derivative
            if order > 1:
                steps.append({
                    "explanation": f"Compute the derivative of order {i}",
                    "expression": f"\\frac{{d^{i}}}{{d{sp.latex(variable)}^{i}}}[{sp.latex(expression)}] = {sp.latex(current_derivative)}"
                })
            else:
                steps.append({
                    "explanation": "Apply the rules of differentiation",
                    "expression": f"\\frac{{d}}{{d{sp.latex(variable)}}}[{sp.latex(expression)}] = {sp.latex(current_derivative)}"
                })
            
            current_expr = current_derivative
        
        return steps
    
    def _generate_integration_steps(self, 
                                  expression: sp.Expr, 
                                  variable: sp.Symbol, 
                                  integral: sp.Expr, 
                                  lower_bound: Optional[sp.Expr] = None, 
                                  upper_bound: Optional[sp.Expr] = None) -> List[Dict[str, str]]:
        """
        Generate steps for integration.
        
        Args:
            expression: Original expression
            variable: Variable of integration
            integral: Computed integral
            lower_bound: Lower bound for definite integration (optional)
            upper_bound: Upper bound for definite integration (optional)
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        if lower_bound is not None and upper_bound is not None:
            steps.append({
                "explanation": "Start with the definite integral",
                "expression": f"\\int_{{{sp.latex(lower_bound)}}}^{{{sp.latex(upper_bound)}}} {sp.latex(expression)} \\, d{sp.latex(variable)}"
            })
        else:
            steps.append({
                "explanation": "Start with the indefinite integral",
                "expression": f"\\int {sp.latex(expression)} \\, d{sp.latex(variable)}"
            })
        
        # Step 2: Apply integration rules
        # In a real implementation, we would detect what rules are being applied
        # and provide more specific steps. This is a simplified version.
        
        # Check if it's a basic function
        if expression.is_polynomial(variable):
            steps.append({
                "explanation": "Apply the power rule for integration",
                "expression": f"\\int {sp.latex(variable)}^n \\, d{sp.latex(variable)} = \\frac{{{sp.latex(variable)}^{{n+1}}}}{{n+1}} + C"
            })
        elif expression.has(sp.sin(variable)) or expression.has(sp.cos(variable)):
            steps.append({
                "explanation": "Apply trigonometric integration rules",
                "expression": "\\text{Use standard trigonometric integration formulas}"
            })
        elif expression.has(sp.exp(variable)):
            steps.append({
                "explanation": "Apply the exponential integration rule",
                "expression": f"\\int e^{{{sp.latex(variable)}}} \\, d{sp.latex(variable)} = e^{{{sp.latex(variable)}}} + C"
            })
        else:
            steps.append({
                "explanation": "Apply appropriate integration techniques",
                "expression": "\\text{(Integration steps may involve substitution, parts, or other methods)}"
            })
        
        # Step 3: Result of indefinite integration
        if lower_bound is None or upper_bound is None:
            steps.append({
                "explanation": "The indefinite integral is",
                "expression": f"{sp.latex(integral)} + C"
            })
        else:
            # Step 3a: Show antiderivative for definite integral
            antiderivative = sp.integrate(expression, variable)
            steps.append({
                "explanation": "Find the antiderivative",
                "expression": f"F({sp.latex(variable)}) = {sp.latex(antiderivative)}"
            })
            
            # Step 3b: Evaluate at the bounds
            upper_value = antiderivative.subs(variable, upper_bound)
            lower_value = antiderivative.subs(variable, lower_bound)
            
            steps.append({
                "explanation": "Evaluate the antiderivative at the upper and lower bounds",
                "expression": f"F({sp.latex(upper_bound)}) - F({sp.latex(lower_bound)}) = {sp.latex(upper_value)} - {sp.latex(lower_value)} = {sp.latex(integral)}"
            })
        
        return steps
    
    def _generate_limit_steps(self, 
                            expression: sp.Expr, 
                            variable: sp.Symbol, 
                            point: sp.Expr, 
                            limit: sp.Expr, 
                            dir_symbol: str) -> List[Dict[str, str]]:
        """
        Generate steps for limit computation.
        
        Args:
            expression: Original expression
            variable: Limit variable
            point: Point of approach
            limit: Computed limit
            dir_symbol: Direction symbol ("^+", "^-", or "")
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original limit expression
        steps.append({
            "explanation": "Start with the limit expression",
            "expression": f"\\lim_{{{sp.latex(variable)} \\to {sp.latex(point)}{dir_symbol}}} {sp.latex(expression)}"
        })
        
        # Step 2: Check for simple substitution
        try:
            direct_sub = expression.subs(variable, point)
            if not direct_sub.has(sp.nan) and not direct_sub.has(sp.oo) and not direct_sub.has(-sp.oo):
                steps.append({
                    "explanation": "Try direct substitution of the limit value",
                    "expression": f"{sp.latex(expression)}|_{{{sp.latex(variable)}={sp.latex(point)}}} = {sp.latex(direct_sub)}"
                })
                
                if direct_sub == limit:
                    steps.append({
                        "explanation": "Direct substitution yields the limit",
                        "expression": f"\\lim_{{{sp.latex(variable)} \\to {sp.latex(point)}{dir_symbol}}} {sp.latex(expression)} = {sp.latex(limit)}"
                    })
                    return steps
            else:
                steps.append({
                    "explanation": "Direct substitution leads to an indeterminate form",
                    "expression": f"{sp.latex(expression)}|_{{{sp.latex(variable)}={sp.latex(point)}}} = \\text{{indeterminate}}"
                })
        except:
            steps.append({
                "explanation": "Direct substitution is not applicable",
                "expression": "\\text{(Need to use limit techniques)}"
            })
        
        # Step 3: Apply techniques based on expression type
        # This is a simplified implementation
        if expression.is_rational_function(variable):
            steps.append({
                "explanation": "For rational functions, factor the numerator and denominator",
                "expression": "\\text{(Factorization steps omitted)}"
            })
        elif expression.has(sp.sin(variable)) or expression.has(sp.cos(variable)):
            steps.append({
                "explanation": "For trigonometric limits, special limit formulas may apply",
                "expression": "\\text{e.g., } \\lim_{x \\to 0} \\frac{\\sin x}{x} = 1"
            })
        else:
            steps.append({
                "explanation": "Apply appropriate limit techniques",
                "expression": "\\text{(L'Hôpital's rule, algebraic manipulation, etc.)}"
            })
        
        # Step 4: Final limit value
        steps.append({
            "explanation": "The limit evaluates to",
            "expression": f"\\lim_{{{sp.latex(variable)} \\to {sp.latex(point)}{dir_symbol}}} {sp.latex(expression)} = {sp.latex(limit)}"
        })
        
        return steps
    
    def _generate_series_steps(self, 
                             expression: sp.Expr, 
                             variable: sp.Symbol, 
                             point: sp.Expr, 
                             order: int, 
                             series: sp.Expr) -> List[Dict[str, str]]:
        """
        Generate steps for series expansion.
        
        Args:
            expression: Original expression
            variable: Series variable
            point: Expansion point
            order: Expansion order
            series: Computed series
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        if point == 0:
            steps.append({
                "explanation": f"Find the Maclaurin series expansion of order {order}",
                "expression": f"{sp.latex(expression)}"
            })
        else:
            steps.append({
                "explanation": f"Find the Taylor series expansion of order {order} around {sp.latex(point)}",
                "expression": f"{sp.latex(expression)}"
            })
        
        # Step 2: Taylor series formula
        steps.append({
            "explanation": "The Taylor series formula is",
            "expression": f"f({sp.latex(variable)}) = \\sum_{{n=0}}^{{\\infty}} \\frac{{f^{{(n)}}({sp.latex(point)})}}{{n!}}({sp.latex(variable)}-{sp.latex(point)})^n"
        })
        
        # Step 3: Compute derivatives at the point
        # In a real implementation, we would show the derivatives
        # This is a simplified version
        steps.append({
            "explanation": "Compute the derivatives at the expansion point",
            "expression": "f({sp.latex(point)}), f'({sp.latex(point)}), f''({sp.latex(point)}), \\ldots"
        })
        
        # Step 4: Final series expansion
        steps.append({
            "explanation": f"The series expansion up to order {order} is",
            "expression": f"{sp.latex(expression)} = {sp.latex(series)} + O(({sp.latex(variable)}-{sp.latex(point)})^{{{order+1}}})"
        })
        
        return steps

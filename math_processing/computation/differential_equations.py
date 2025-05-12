"""
Differential equation solver with support for various solution methods.

This module provides functionality for solving differential equations,
with emphasis on first-order linear differential equations using methods
like integrating factor.
"""
import sympy as sp
import numpy as np
import logging
from typing import Union, Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DifferentialEquationSolver:
    """Solver for differential equations."""
    
    def __init__(self):
        """Initialize the differential equation solver."""
        pass
    
    def solve_first_order_linear(self, 
                                equation: Union[sp.Eq, str], 
                                dependent_var: Union[sp.Symbol, str] = None,
                                independent_var: Union[sp.Symbol, str] = None) -> Dict[str, Any]:
        """
        Solve a first-order linear differential equation of the form dy/dx + P(x)y = Q(x).
        
        Args:
            equation: The differential equation
            dependent_var: The dependent variable (default: y)
            independent_var: The independent variable (default: x)
            
        Returns:
            Dictionary containing solution and steps
        """
        try:
            # Process variables
            if dependent_var is None:
                dependent_var = sp.Symbol('y')
            elif isinstance(dependent_var, str):
                dependent_var = sp.Symbol(dependent_var)
                
            if independent_var is None:
                independent_var = sp.Symbol('x')
            elif isinstance(independent_var, str):
                independent_var = sp.Symbol(independent_var)
                
            # Parse equation if needed
            if isinstance(equation, str):
                # Check if it's in the form dy/dx = f(x,y)
                if 'dy/dx' in equation or 'd' + str(dependent_var) + '/d' + str(independent_var) in equation:
                    equation = equation.replace('dy/dx', 'diff(y, x)')
                    equation = equation.replace(f'd{dependent_var}/d{independent_var}', f'diff({dependent_var}, {independent_var})')
                equation = sp.sympify(equation)
                
            # Convert equation to standard form: dy/dx + P(x)y = Q(x)
            if isinstance(equation, sp.Eq):
                # Move all terms to one side
                eq_rhs = equation.rhs
                eq_lhs = equation.lhs
                
                # Check if one side is dy/dx or diff(y, x)
                if eq_lhs == sp.diff(dependent_var, independent_var):
                    # Form: dy/dx = f(x,y)
                    # Transform to: dy/dx - f(x,y) = 0
                    eq_expr = eq_lhs - eq_rhs
                elif eq_rhs == sp.diff(dependent_var, independent_var):
                    # Form: f(x,y) = dy/dx
                    # Transform to: f(x,y) - dy/dx = 0
                    eq_expr = eq_lhs - eq_rhs
                else:
                    # Assume the equation is already in the proper form
                    eq_expr = equation
            else:
                # Assume equation is already in the form dy/dx + P(x)y = Q(x)
                eq_expr = equation
            
            # Extract P(x) and Q(x) from the equation
            # We need to convert to the form dy/dx + P(x)y = Q(x)
            derivative_term = sp.diff(dependent_var, independent_var)
            
            # Collect terms with the derivative
            derivative_coef = sp.collect(eq_expr, derivative_term).coeff(derivative_term)
            remaining_expr = eq_expr - derivative_coef * derivative_term
            
            # If the derivative coefficient is not 1, divide by it
            if derivative_coef != 1:
                remaining_expr = remaining_expr / derivative_coef
                
            # Collect terms with the dependent variable
            y_coef = sp.collect(remaining_expr, dependent_var).coeff(dependent_var)
            const_term = remaining_expr - y_coef * dependent_var
            
            # Now we have: dy/dx + y_coef * y = const_term
            P_x = y_coef
            Q_x = -const_term  # Note: our form has it on the right side
            
            # Compute integrating factor: e^∫P(x)dx
            integrating_factor_exponent = sp.integrate(P_x, independent_var)
            integrating_factor = sp.exp(integrating_factor_exponent)
            
            # Multiply both sides by the integrating factor
            lhs_with_if = integrating_factor * derivative_term + integrating_factor * P_x * dependent_var
            rhs_with_if = integrating_factor * Q_x
            
            # Recognize the left side as a product rule derivative: d/dx[e^(∫P dx) * y]
            product_derivative = sp.diff(integrating_factor * dependent_var, independent_var)
            
            # Integrate both sides
            integrated_rhs = sp.integrate(rhs_with_if, independent_var)
            
            # Solve for y
            general_solution = integrated_rhs / integrating_factor + sp.Symbol('C') / integrating_factor
            
            # Generate solution steps
            steps = self._generate_integrating_factor_steps(
                eq_expr, 
                P_x, 
                Q_x, 
                integrating_factor_exponent,
                integrating_factor,
                integrated_rhs,
                general_solution,
                dependent_var,
                independent_var
            )
            
            return {
                "success": True,
                "method": "integrating_factor",
                "solution": {
                    "general_solution": general_solution,
                    "particular_solution": None  # Would require initial conditions
                },
                "steps": steps
            }
            
        except Exception as e:
            logger.error(f"Error solving first-order linear ODE: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_integrating_factor_steps(self,
                                         equation: sp.Expr,
                                         P_x: sp.Expr,
                                         Q_x: sp.Expr,
                                         if_exponent: sp.Expr,
                                         integrating_factor: sp.Expr,
                                         integrated_rhs: sp.Expr,
                                         general_solution: sp.Expr,
                                         y: sp.Symbol,
                                         x: sp.Symbol) -> List[Dict[str, str]]:
        """
        Generate steps for solving a first-order linear ODE using the integrating factor method.
        
        Args:
            equation: The differential equation
            P_x: The coefficient of y
            Q_x: The right-hand side
            if_exponent: The exponent of the integrating factor
            integrating_factor: The integrating factor
            integrated_rhs: The right-hand side after integration
            general_solution: The general solution
            y: The dependent variable symbol
            x: The independent variable symbol
            
        Returns:
            List of solution steps as dictionaries
        """
        steps = []
        
        # Step 1: Identify the equation as a first-order linear ODE
        steps.append({
            "step": 1,
            "type": "identify",
            "explanation": "Identify the equation as a first-order linear differential equation",
            "latex": sp.latex(sp.Eq(sp.diff(y, x) + P_x * y, Q_x))
        })
        
        # Step 2: Write it in standard form
        steps.append({
            "step": 2,
            "type": "standard_form",
            "explanation": "Write the equation in standard form: dy/dx + P(x)y = Q(x)",
            "latex": sp.latex(sp.Eq(sp.diff(y, x) + P_x * y, Q_x))
        })
        
        # Step 3: Identify P(x) and Q(x)
        steps.append({
            "step": 3,
            "type": "identify_functions",
            "explanation": "Identify P(x) and Q(x) in the standard form",
            "latex": f"P(x) = {sp.latex(P_x)}, Q(x) = {sp.latex(Q_x)}"
        })
        
        # Step 4: Calculate the integrating factor
        steps.append({
            "step": 4,
            "type": "integrating_factor",
            "explanation": "Calculate the integrating factor: e^∫P(x)dx",
            "latex": f"\\mu(x) = e^{{\\int {sp.latex(P_x)}\\,dx}} = e^{{{sp.latex(if_exponent)}}} = {sp.latex(integrating_factor)}"
        })
        
        # Step 5: Multiply the equation by the integrating factor
        steps.append({
            "step": 5,
            "type": "multiply_by_if",
            "explanation": "Multiply both sides of the equation by the integrating factor",
            "latex": sp.latex(sp.Eq(integrating_factor * sp.diff(y, x) + integrating_factor * P_x * y, integrating_factor * Q_x))
        })
        
        # Step 6: Recognize the left side as a product rule derivative
        steps.append({
            "step": 6,
            "type": "recognize_pattern",
            "explanation": "Recognize the left side as a product rule derivative: d/dx[μ(x)y]",
            "latex": sp.latex(sp.Eq(sp.diff(integrating_factor * y, x), integrating_factor * Q_x))
        })
        
        # Step 7: Integrate both sides
        steps.append({
            "step": 7,
            "type": "integrate",
            "explanation": "Integrate both sides with respect to x",
            "latex": sp.latex(sp.Eq(integrating_factor * y, integrated_rhs + sp.Symbol('C')))
        })
        
        # Step 8: Solve for y
        steps.append({
            "step": 8,
            "type": "solve_for_y",
            "explanation": "Solve for y by dividing both sides by the integrating factor",
            "latex": sp.latex(sp.Eq(y, general_solution))
        })
        
        # Step 9: Final general solution
        steps.append({
            "step": 9,
            "type": "general_solution",
            "explanation": "The general solution is",
            "latex": sp.latex(sp.Eq(y, general_solution))
        })
        
        return steps
    
    def solve_differential_equation(self,
                                   equation: Union[sp.Eq, str],
                                   dependent_var: Union[sp.Symbol, str] = None,
                                   independent_var: Union[sp.Symbol, str] = None,
                                   initial_conditions: Optional[Dict[float, float]] = None,
                                   method: Optional[str] = None) -> Dict[str, Any]:
        """
        General method to solve differential equations.
        
        Args:
            equation: The differential equation
            dependent_var: The dependent variable
            independent_var: The independent variable
            initial_conditions: Dictionary of initial conditions {x_0: y_0}
            method: Solution method (auto, integrating_factor, separation_of_variables, etc.)
            
        Returns:
            Dictionary containing solution and steps
        """
        try:
            # Process variables
            if dependent_var is None:
                dependent_var = sp.Symbol('y')
            elif isinstance(dependent_var, str):
                dependent_var = sp.Symbol(dependent_var)
                
            if independent_var is None:
                independent_var = sp.Symbol('x')
            elif isinstance(independent_var, str):
                independent_var = sp.Symbol(independent_var)
            
            # Auto-detect method if not specified
            if method is None or method == "auto":
                # TODO: Implement method detection logic
                # For now, default to first-order linear solver
                method = "integrating_factor"
            
            # Solve based on detected method
            if method == "integrating_factor":
                result = self.solve_first_order_linear(equation, dependent_var, independent_var)
                
                # Apply initial conditions if provided
                if initial_conditions and result["success"]:
                    # TODO: Apply initial conditions to the general solution
                    pass
                
                return result
            else:
                return {
                    "success": False,
                    "error": f"Unsupported method: {method}"
                }
                
        except Exception as e:
            logger.error(f"Error solving differential equation: {e}")
            return {
                "success": False,
                "error": str(e)
            } 
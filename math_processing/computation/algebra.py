"""
Algebra Module - specialized algebraic operations using SymPy.

This module implements algebraic operations including equation solving,
polynomial manipulation, and symbolic simplification.
"""

import sympy as sp
from typing import Dict, List, Union, Any, Optional, Tuple
import logging


class AlgebraProcessor:
    """Processor for algebraic operations."""
    
    def __init__(self):
        """Initialize the algebra processor."""
        self.logger = logging.getLogger(__name__)
    
    def solve_polynomial(self, 
                        polynomial: Union[sp.Expr, str], 
                        variable: Union[sp.Symbol, str]) -> Dict[str, Any]:
        """
        Solve a polynomial equation set equal to zero.
        
        Args:
            polynomial: Polynomial expression
            variable: Variable to solve for
            
        Returns:
            Dictionary with solution information
        """
        try:
            # Handle string input
            if isinstance(polynomial, str):
                polynomial = sp.sympify(polynomial)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Create an equation: polynomial = 0
            equation = sp.Eq(polynomial, 0)
            
            # Solve the equation
            solutions = sp.solve(equation, variable)
            
            # Classify solutions
            real_solutions = []
            complex_solutions = []
            
            for sol in solutions:
                if sp.im(sol) == 0:
                    real_solutions.append(sol)
                else:
                    complex_solutions.append(sol)
            
            # Generate steps for solving
            steps = self._generate_polynomial_steps(polynomial, variable, solutions)
            
            return {
                "success": True,
                "solutions": solutions,
                "real_solutions": real_solutions,
                "complex_solutions": complex_solutions,
                "steps": steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error solving polynomial: {str(e)}")
            return {
                "success": False,
                "solutions": None,
                "real_solutions": None,
                "complex_solutions": None,
                "steps": None,
                "error": str(e)
            }
    
    def factor_expression(self, 
                         expression: Union[sp.Expr, str],
                         advanced: bool = False) -> Dict[str, Any]:
        """
        Factor a mathematical expression.
        
        Args:
            expression: Mathematical expression to factor
            advanced: Whether to use advanced factoring techniques
            
        Returns:
            Dictionary with factorization information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Apply factoring
            factored = sp.factor(expression)
            
            # For advanced factoring, try additional methods
            if advanced:
                # Try factoring over complex domain
                complex_factored = sp.factor(expression, extension=sp.I)
                
                # Check if complex factoring gives a different result
                if complex_factored != factored:
                    return {
                        "success": True,
                        "factored": factored,
                        "complex_factored": complex_factored,
                        "steps": self._generate_factoring_steps(expression, factored, complex_factored),
                        "error": None
                    }
            
            return {
                "success": True,
                "factored": factored,
                "steps": self._generate_factoring_steps(expression, factored),
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error factoring expression: {str(e)}")
            return {
                "success": False,
                "factored": None,
                "steps": None,
                "error": str(e)
            }
    
    def solve_system(self, 
                    equations: List[Union[sp.Eq, str]], 
                    variables: List[Union[sp.Symbol, str]]) -> Dict[str, Any]:
        """
        Solve a system of algebraic equations.
        
        Args:
            equations: List of equations
            variables: List of variables to solve for
            
        Returns:
            Dictionary with solution information
        """
        try:
            # Process equations
            processed_equations = []
            for eq in equations:
                if isinstance(eq, str):
                    if '=' in eq:
                        lhs, rhs = eq.split('=', 1)
                        processed_equations.append(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)))
                    else:
                        # If no equals sign, assume equation = 0
                        processed_equations.append(sp.Eq(sp.sympify(eq), 0))
                else:
                    processed_equations.append(eq)
            
            # Process variables
            processed_variables = []
            for var in variables:
                if isinstance(var, str):
                    processed_variables.append(sp.Symbol(var))
                else:
                    processed_variables.append(var)
            
            # Solve the system
            solution = sp.solve(processed_equations, processed_variables, dict=True)
            
            # Generate steps
            steps = self._generate_system_steps(processed_equations, processed_variables, solution)
            
            # Handle the case where solution is a list
            if isinstance(solution, list):
                # Multiple solution sets
                return {
                    "success": True,
                    "solutions": solution,
                    "steps": steps,
                    "error": None
                }
            else:
                # Single solution set
                return {
                    "success": True,
                    "solutions": [solution] if solution else [],
                    "steps": steps,
                    "error": None
                }
        except Exception as e:
            self.logger.error(f"Error solving system: {str(e)}")
            return {
                "success": False,
                "solutions": None,
                "steps": None,
                "error": str(e)
            }
    
    def complete_square(self, 
                       expression: Union[sp.Expr, str], 
                       variable: Union[sp.Symbol, str]) -> Dict[str, Any]:
        """
        Complete the square for a quadratic expression.
        
        Args:
            expression: Quadratic expression
            variable: Variable to complete the square with respect to
            
        Returns:
            Dictionary with the result of completing the square
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Handle string variable
            if isinstance(variable, str):
                variable = sp.Symbol(variable)
            
            # Extract the coefficients of x^2, x, and constant term
            poly = sp.Poly(expression, variable)
            if poly.degree() != 2:
                return {
                    "success": False,
                    "completed_form": None,
                    "steps": None,
                    "error": "Expression must be a quadratic in the specified variable"
                }
            
            a, b, c = poly.all_coeffs()
            
            # Complete the square: a(x + b/(2a))^2 + c - b^2/(4a)
            b_over_2a = b/(2*a)
            perfect_square = a * (variable + b_over_2a)**2
            constant_term = c - (b**2)/(4*a)
            completed_form = perfect_square + constant_term
            
            # Generate steps
            steps = self._generate_complete_square_steps(expression, variable, a, b, c, completed_form)
            
            return {
                "success": True,
                "completed_form": completed_form,
                "a": a,
                "b": b,
                "c": c,
                "steps": steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error completing square: {str(e)}")
            return {
                "success": False,
                "completed_form": None,
                "steps": None,
                "error": str(e)
            }
    
    def simplify_rational(self, 
                         expression: Union[sp.Expr, str]) -> Dict[str, Any]:
        """
        Simplify a rational expression by factoring numerator and denominator.
        
        Args:
            expression: Rational expression to simplify
            
        Returns:
            Dictionary with simplification information
        """
        try:
            # Handle string input
            if isinstance(expression, str):
                expression = sp.sympify(expression)
            
            # Check if the expression is a fraction
            if expression.is_rational_function():
                # Get numerator and denominator
                num, den = expression.as_numer_denom()
                
                # Factor numerator and denominator
                num_factored = sp.factor(num)
                den_factored = sp.factor(den)
                
                # Simplify the fraction
                simplified = sp.cancel(expression)
                
                # Generate steps
                steps = self._generate_rational_steps(expression, num, den, num_factored, den_factored, simplified)
                
                return {
                    "success": True,
                    "simplified": simplified,
                    "numerator": num,
                    "denominator": den,
                    "numerator_factored": num_factored,
                    "denominator_factored": den_factored,
                    "steps": steps,
                    "error": None
                }
            else:
                # Not a rational function, just try to simplify
                simplified = sp.simplify(expression)
                
                return {
                    "success": True,
                    "simplified": simplified,
                    "numerator": expression,
                    "denominator": 1,
                    "numerator_factored": sp.factor(expression),
                    "denominator_factored": 1,
                    "steps": [{"explanation": "Expression is not a fraction, applying general simplification", 
                               "expression": simplified}],
                    "error": None
                }
        except Exception as e:
            self.logger.error(f"Error simplifying rational: {str(e)}")
            return {
                "success": False,
                "simplified": None,
                "numerator": None,
                "denominator": None,
                "numerator_factored": None,
                "denominator_factored": None,
                "steps": None,
                "error": str(e)
            }
    
    def _generate_polynomial_steps(self, 
                                  polynomial: sp.Expr, 
                                  variable: sp.Symbol, 
                                  solutions: List[sp.Expr]) -> List[Dict[str, str]]:
        """
        Generate steps for solving a polynomial equation.
        
        Args:
            polynomial: Polynomial expression
            variable: Variable being solved for
            solutions: List of solutions
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Write the equation
        equation = sp.Eq(polynomial, 0)
        steps.append({
            "explanation": "Set up the equation by setting the polynomial equal to zero",
            "expression": sp.latex(equation)
        })
        
        # Check the degree of the polynomial
        degree = sp.degree(polynomial, gen=variable)
        
        if degree == 1:
            # Step 2: Solve linear equation
            steps.append({
                "explanation": "Solve the linear equation for " + sp.latex(variable),
                "expression": sp.latex(variable) + " = " + sp.latex(solutions[0])
            })
        
        elif degree == 2:
            # Step 2: Identify as quadratic
            a, b, c = sp.poly(polynomial, variable).all_coeffs()
            steps.append({
                "explanation": f"Identify the quadratic equation in the form a{sp.latex(variable)}^2 + b{sp.latex(variable)} + c = 0",
                "expression": f"a = {sp.latex(a)}, b = {sp.latex(b)}, c = {sp.latex(c)}"
            })
            
            # Step 3: Calculate discriminant
            discriminant = b**2 - 4*a*c
            steps.append({
                "explanation": "Calculate the discriminant b^2 - 4ac",
                "expression": f"\\Delta = {sp.latex(b)}^2 - 4 \\cdot {sp.latex(a)} \\cdot {sp.latex(c)} = {sp.latex(discriminant)}"
            })
            
            # Step 4: Apply quadratic formula
            steps.append({
                "explanation": "Apply the quadratic formula " + sp.latex(variable) + " = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
                "expression": sp.latex(variable) + " = \\frac{" + sp.latex(-b) + " \\pm \\sqrt{" + sp.latex(discriminant) + "}}{" + sp.latex(2*a) + "}"
            })
            
            # Step 5: Final solutions
            solutions_latex = ", ".join([sp.latex(sol) for sol in solutions])
            steps.append({
                "explanation": "The solutions are",
                "expression": sp.latex(variable) + " = " + solutions_latex
            })
        
        elif degree == 3 or degree == 4:
            # For cubic and quartic equations, we don't show the full solution process
            # because it's quite complex, but we acknowledge the approach
            steps.append({
                "explanation": f"For this {'cubic' if degree == 3 else 'quartic'} equation, we use specialized formulas to find the roots",
                "expression": "\\text{(Solution process involves complex algebraic formulas)}"
            })
            
            # Final solutions
            solutions_latex = ", ".join([sp.latex(sol) for sol in solutions])
            steps.append({
                "explanation": "The solutions are",
                "expression": sp.latex(variable) + " = " + solutions_latex
            })
        
        else:
            # For higher degree polynomials
            steps.append({
                "explanation": "For this higher-degree polynomial, numerical methods are used to find the roots",
                "expression": "\\text{(Numerical approximation methods applied)}"
            })
            
            # Final solutions
            solutions_latex = ", ".join([sp.latex(sol) for sol in solutions])
            steps.append({
                "explanation": "The solutions are",
                "expression": sp.latex(variable) + " = " + solutions_latex
            })
        
        return steps
    
    def _generate_factoring_steps(self, 
                                expression: sp.Expr, 
                                factored: sp.Expr, 
                                complex_factored: Optional[sp.Expr] = None) -> List[Dict[str, str]]:
        """
        Generate steps for factoring an expression.
        
        Args:
            expression: Original expression
            factored: Factored expression
            complex_factored: Expression factored over complex field (optional)
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        steps.append({
            "explanation": "Start with the original expression",
            "expression": sp.latex(expression)
        })
        
        # Step 2: Factor over the reals
        steps.append({
            "explanation": "Factor the expression over the real numbers",
            "expression": sp.latex(factored)
        })
        
        # Step 3: Factor over complex numbers (if provided and different)
        if complex_factored is not None and complex_factored != factored:
            steps.append({
                "explanation": "Further factorization is possible over the complex numbers",
                "expression": sp.latex(complex_factored)
            })
        
        return steps
    
    def _generate_system_steps(self, 
                             equations: List[sp.Eq], 
                             variables: List[sp.Symbol], 
                             solution: List[Dict[sp.Symbol, sp.Expr]]) -> List[Dict[str, str]]:
        """
        Generate steps for solving a system of equations.
        
        Args:
            equations: List of equations
            variables: List of variables
            solution: Solution dictionary or list of dictionaries
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original system
        system_latex = "\\begin{cases} "
        for eq in equations:
            system_latex += sp.latex(eq) + " \\\\ "
        system_latex += "\\end{cases}"
        
        steps.append({
            "explanation": "Start with the system of equations",
            "expression": system_latex
        })
        
        # If there are only two equations with two unknowns, show elimination method
        if len(equations) == 2 and len(variables) == 2:
            # Step 2: Solve for one variable from the first equation
            # This is a simplified approach - in a real implementation, 
            # you would need to check if this is possible and maybe try different approaches
            try:
                # Try to solve the first equation for the first variable
                var1_expr = sp.solve(equations[0], variables[0])[0]
                
                steps.append({
                    "explanation": f"Solve the first equation for {sp.latex(variables[0])}",
                    "expression": sp.latex(variables[0]) + " = " + sp.latex(var1_expr)
                })
                
                # Step 3: Substitute into the second equation
                eq2_subst = equations[1].subs(variables[0], var1_expr)
                
                steps.append({
                    "explanation": f"Substitute this expression for {sp.latex(variables[0])} into the second equation",
                    "expression": sp.latex(eq2_subst)
                })
                
                # Step 4: Solve the resulting equation for the second variable
                var2_value = sp.solve(eq2_subst, variables[1])[0]
                
                steps.append({
                    "explanation": f"Solve for {sp.latex(variables[1])}",
                    "expression": sp.latex(variables[1]) + " = " + sp.latex(var2_value)
                })
                
                # Step 5: Find the value of the first variable
                var1_value = var1_expr.subs(variables[1], var2_value)
                
                steps.append({
                    "explanation": f"Substitute the value of {sp.latex(variables[1])} back to find {sp.latex(variables[0])}",
                    "expression": sp.latex(variables[0]) + " = " + sp.latex(var1_value)
                })
                
            except Exception:
                # If the elimination method fails, just show a generic message
                steps.append({
                    "explanation": "Solve the system using elimination or substitution methods",
                    "expression": "\\text{(Detailed solution steps omitted)}"
                })
        else:
            # For larger systems, just show a generic message
            steps.append({
                "explanation": "Apply elimination or matrix methods to solve the system",
                "expression": "\\text{(Detailed solution steps omitted)}"
            })
        
        # Final step: Show solution
        if solution:
            if isinstance(solution, list):
                # Multiple solution sets
                if len(solution) == 1:
                    # Single solution set as a list
                    solution_latex = ", ".join([sp.latex(var) + " = " + sp.latex(solution[0][var]) for var in variables])
                    steps.append({
                        "explanation": "The solution to the system is",
                        "expression": solution_latex
                    })
                else:
                    # Multiple solution sets
                    solution_latex = "\\text{Solution Set 1: }"
                    solution_latex += ", ".join([sp.latex(var) + " = " + sp.latex(solution[0][var]) for var in variables])
                    for i, sol in enumerate(solution[1:], 2):
                        solution_latex += "\\\\ \\text{Solution Set " + str(i) + ": }"
                        solution_latex += ", ".join([sp.latex(var) + " = " + sp.latex(sol[var]) for var in variables])
                    
                    steps.append({
                        "explanation": "The system has multiple solution sets",
                        "expression": solution_latex
                    })
            else:
                # Single solution set as a dictionary
                solution_latex = ", ".join([sp.latex(var) + " = " + sp.latex(solution[var]) for var in variables])
                steps.append({
                    "explanation": "The solution to the system is",
                    "expression": solution_latex
                })
        else:
            # No solution or infinite solutions
            steps.append({
                "explanation": "The system has no solution or infinitely many solutions",
                "expression": "\\text{No specific solution values to report}"
            })
        
        return steps
    
    def _generate_complete_square_steps(self, 
                                      expression: sp.Expr, 
                                      variable: sp.Symbol, 
                                      a: sp.Expr, 
                                      b: sp.Expr, 
                                      c: sp.Expr, 
                                      completed_form: sp.Expr) -> List[Dict[str, str]]:
        """
        Generate steps for completing the square.
        
        Args:
            expression: Original expression
            variable: Variable
            a: Coefficient of x^2
            b: Coefficient of x
            c: Constant term
            completed_form: Expression after completing the square
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        steps.append({
            "explanation": "Start with the quadratic expression",
            "expression": sp.latex(expression)
        })
        
        # Step 2: Identify coefficients
        steps.append({
            "explanation": f"Identify the coefficients in the form a{sp.latex(variable)}^2 + b{sp.latex(variable)} + c",
            "expression": f"a = {sp.latex(a)}, b = {sp.latex(b)}, c = {sp.latex(c)}"
        })
        
        # Step 3: Factor out the leading coefficient if not 1
        if a != 1:
            factored_expr = a * (variable**2 + b/a * variable + c/a)
            steps.append({
                "explanation": f"Factor out the leading coefficient {sp.latex(a)}",
                "expression": sp.latex(factored_expr)
            })
        else:
            factored_expr = expression
        
        # Step 4: Set up for completing the square
        if a != 1:
            b_over_a = b/a
            partial_expr = variable**2 + b_over_a * variable
            steps.append({
                "explanation": f"Focus on the terms with the variable inside the parentheses: {sp.latex(variable)}^2 + {sp.latex(b_over_a)}{sp.latex(variable)}",
                "expression": sp.latex(a) + "(" + sp.latex(partial_expr) + " + " + sp.latex(c/a) + ")"
            })
        else:
            partial_expr = variable**2 + b * variable
            steps.append({
                "explanation": f"Focus on the terms with the variable: {sp.latex(partial_expr)}",
                "expression": sp.latex(partial_expr) + " + " + sp.latex(c)
            })
        
        # Step 5: Find the value to add and subtract
        if a != 1:
            b_over_2a = b/(2*a)
            perfect_square_term = b_over_2a**2
            steps.append({
                "explanation": f"To complete the square, add and subtract ({sp.latex(b_over_2a)})^2 = {sp.latex(perfect_square_term)}",
                "expression": sp.latex(a) + "(" + sp.latex(partial_expr) + " + " + sp.latex(perfect_square_term) + " - " + sp.latex(perfect_square_term) + " + " + sp.latex(c/a) + ")"
            })
        else:
            b_over_2 = b/2
            perfect_square_term = b_over_2**2
            steps.append({
                "explanation": f"To complete the square, add and subtract ({sp.latex(b_over_2)})^2 = {sp.latex(perfect_square_term)}",
                "expression": sp.latex(partial_expr) + " + " + sp.latex(perfect_square_term) + " - " + sp.latex(perfect_square_term) + " + " + sp.latex(c)
            })
        
        # Step 6: Create the perfect square trinomial
        if a != 1:
            b_over_2a = b/(2*a)
            perfect_square = (variable + b_over_2a)**2
            constant_term = c/a - b_over_2a**2
            steps.append({
                "explanation": f"Recognize the perfect square trinomial",
                "expression": sp.latex(a) + "(" + sp.latex(perfect_square) + " + " + sp.latex(constant_term) + ")"
            })
        else:
            b_over_2 = b/2
            perfect_square = (variable + b_over_2)**2
            constant_term = c - b_over_2**2
            steps.append({
                "explanation": f"Recognize the perfect square trinomial",
                "expression": sp.latex(perfect_square) + " + " + sp.latex(constant_term)
            })
        
        # Step 7: Final form
        steps.append({
            "explanation": "The expression in completed square form is",
            "expression": sp.latex(completed_form)
        })
        
        return steps
    
    def _generate_rational_steps(self, 
                               expression: sp.Expr, 
                               num: sp.Expr, 
                               den: sp.Expr, 
                               num_factored: sp.Expr, 
                               den_factored: sp.Expr, 
                               simplified: sp.Expr) -> List[Dict[str, str]]:
        """
        Generate steps for simplifying a rational expression.
        
        Args:
            expression: Original rational expression
            num: Numerator
            den: Denominator
            num_factored: Factored numerator
            den_factored: Factored denominator
            simplified: Final simplified expression
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Original expression
        steps.append({
            "explanation": "Start with the rational expression",
            "expression": sp.latex(expression)
        })
        
        # Step 2: Separate numerator and denominator
        steps.append({
            "explanation": "Identify the numerator and denominator",
            "expression": "\\frac{" + sp.latex(num) + "}{" + sp.latex(den) + "}"
        })
        
        # Step 3: Factor numerator and denominator
        if num_factored != num or den_factored != den:
            steps.append({
                "explanation": "Factor the numerator and denominator",
                "expression": "\\frac{" + sp.latex(num_factored) + "}{" + sp.latex(den_factored) + "}"
            })
        
        # Step 4: Cancel common factors
        if simplified != expression:
            steps.append({
                "explanation": "Cancel common factors",
                "expression": sp.latex(simplified)
            })
        else:
            steps.append({
                "explanation": "No common factors to cancel",
                "expression": sp.latex(simplified)
            })
        
        return steps

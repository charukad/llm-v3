"""
Step-by-Step Solution Generator for Mathematical Problems

This module provides comprehensive step-by-step solution generation for various
mathematical problems across different domains including algebra, calculus,
and linear algebra. The solution steps are generated with educational explanations
suitable for learning and teaching contexts.
"""

import sympy as sp
from typing import List, Dict, Any, Union, Optional, Tuple
import logging
from math_processing.expressions.converters import sympy_to_latex
from math_processing.computation.sympy_wrapper import SymbolicProcessor
from math_processing.knowledge.knowledge_base import MathKnowledgeBase

# Configure logging
logger = logging.getLogger(__name__)

class SolutionStep:
    """Represents a single step in a mathematical solution."""
    
    def __init__(
        self, 
        step_number: int,
        operation: str,
        input_expr: Union[sp.Expr, sp.Eq, str],
        output_expr: Union[sp.Expr, sp.Eq, str],
        explanation: str,
        hint: Optional[str] = None
    ):
        """
        Initialize a solution step.
        
        Args:
            step_number: Position in the solution sequence
            operation: Type of mathematical operation performed
            input_expr: Input expression for this step
            output_expr: Result after applying the operation
            explanation: Natural language explanation of the step
            hint: Optional hint for educational purposes
        """
        self.step_number = step_number
        self.operation = operation
        self.input_expr = input_expr
        self.output_expr = output_expr
        self.explanation = explanation
        self.hint = hint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the step to a dictionary representation."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "input": str(self.input_expr),
            "output": str(self.output_expr),
            "input_latex": sympy_to_latex(self.input_expr) if hasattr(self.input_expr, "free_symbols") else str(self.input_expr),
            "output_latex": sympy_to_latex(self.output_expr) if hasattr(self.output_expr, "free_symbols") else str(self.output_expr),
            "explanation": self.explanation,
            "hint": self.hint
        }


class SolutionGenerator:
    """
    Generates step-by-step solutions for mathematical problems.
    
    This class provides methods to generate detailed solution steps for
    various types of mathematical problems, with domain-specific approaches.
    """
    
    def __init__(self, knowledge_base: Optional[MathKnowledgeBase] = None):
        """
        Initialize the solution generator.
        
        Args:
            knowledge_base: Optional knowledge base for concept references
        """
        self.symbolic_processor = SymbolicProcessor()
        self.knowledge_base = knowledge_base
        
    def generate_solution_steps(
        self, 
        expression: Union[sp.Expr, sp.Eq, str],
        operation: str,
        variables: Optional[List[Union[sp.Symbol, str]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SolutionStep]:
        """
        Generate step-by-step solution for a mathematical operation.
        
        Args:
            expression: The mathematical expression to operate on
            operation: Type of operation to perform (solve, differentiate, integrate, etc.)
            variables: Variables involved in the operation
            context: Additional context for solution generation
            
        Returns:
            List of solution steps
        """
        context = context or {}
        domain = context.get("domain", self._infer_domain(expression, operation))
        
        # Select the appropriate solution generator based on operation and domain
        if operation == "solve":
            return self._generate_solve_steps(expression, variables, domain, context)
        elif operation == "differentiate":
            return self._generate_differentiation_steps(expression, variables, domain, context)
        elif operation == "integrate":
            return self._generate_integration_steps(expression, variables, domain, context)
        elif operation == "limit":
            return self._generate_limit_steps(expression, variables, domain, context)
        elif operation == "matrix_operation":
            return self._generate_matrix_operation_steps(expression, variables, domain, context)
        else:
            logger.warning(f"Unsupported operation: {operation}")
            return []
    
    def _infer_domain(self, expression: Union[sp.Expr, sp.Eq, str], operation: str) -> str:
        """
        Infer the mathematical domain based on expression and operation.
        
        Args:
            expression: The mathematical expression
            operation: Type of operation
            
        Returns:
            Domain string (algebra, calculus, linear_algebra, etc.)
        """
        # Convert string expressions to sympy if needed
        if isinstance(expression, str):
            try:
                expression = sp.sympify(expression)
            except Exception as e:
                logger.warning(f"Failed to sympify expression: {e}")
                # Default to algebra if parsing fails
                return "algebra"
        
        # Check for calculus operations
        if operation in ["differentiate", "integrate", "limit"]:
            return "calculus"
        
        # Check for matrix operations
        if operation == "matrix_operation" or (hasattr(expression, "is_Matrix") and expression.is_Matrix):
            return "linear_algebra"
        
        # Default to algebra for equations and expressions
        return "algebra"
    
    def _generate_solve_steps(
        self,
        equation: Union[sp.Eq, str],
        variables: Optional[List[Union[sp.Symbol, str]]] = None,
        domain: str = "algebra",
        context: Optional[Dict[str, Any]] = None
    ) -> List[SolutionStep]:
        """
        Generate steps for solving an equation or system of equations.
        
        Args:
            equation: The equation to solve
            variables: Variables to solve for
            domain: Mathematical domain
            context: Additional context
            
        Returns:
            List of solution steps
        """
        steps = []
        context = context or {}
        
        # Convert string equation to sympy if needed
        if isinstance(equation, str):
            try:
                # Check if it's already an equation with '=' or just an expression
                if '=' in equation:
                    left, right = equation.split('=', 1)
                    equation = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
                else:
                    # If no equals sign, assume it's being set equal to 0
                    equation = sp.Eq(sp.sympify(equation), 0)
            except Exception as e:
                logger.error(f"Failed to parse equation: {e}")
                return [SolutionStep(
                    1, "error", equation, equation,
                    f"Error parsing equation: {e}"
                )]
        
        # Extract variable if not provided
        if not variables:
            if isinstance(equation, sp.Eq):
                free_symbols = equation.lhs.free_symbols.union(equation.rhs.free_symbols)
                if free_symbols:
                    variables = [list(free_symbols)[0]]
                else:
                    return [SolutionStep(
                        1, "error", equation, equation,
                        "No variables found in equation"
                    )]
        
        # Ensure variables are sympy Symbols
        sym_variables = []
        for var in variables:
            if isinstance(var, str):
                sym_variables.append(sp.Symbol(var))
            else:
                sym_variables.append(var)
        
        # Add initial step showing the original equation
        steps.append(SolutionStep(
            1,
            "initial",
            equation,
            equation,
            f"Starting with the equation: {sympy_to_latex(equation)}"
        ))
        
        # Different solving approaches based on equation type
        if isinstance(equation, sp.Eq):
            if equation.lhs.is_polynomial() and equation.rhs.is_polynomial():
                # Polynomial equation
                return self._solve_polynomial_equation(equation, sym_variables, steps)
            elif any(term.is_rational_function() for term in [equation.lhs, equation.rhs]):
                # Rational equation
                return self._solve_rational_equation(equation, sym_variables, steps)
            else:
                # General approach for other equations
                return self._solve_general_equation(equation, sym_variables, steps)
        
        return steps
    
    def _solve_polynomial_equation(
        self,
        equation: sp.Eq,
        variables: List[sp.Symbol],
        steps: List[SolutionStep]
    ) -> List[SolutionStep]:
        """Handle polynomial equation solving with detailed steps."""
        step_num = len(steps) + 1
        variable = variables[0]  # Use the first variable
        
        # Move everything to one side
        expr = equation.lhs - equation.rhs
        new_equation = sp.Eq(expr, 0)
        
        steps.append(SolutionStep(
            step_num,
            "rearrange",
            equation,
            new_equation,
            f"Rearrange the equation to standard form by moving all terms to the left side: {sympy_to_latex(new_equation)}"
        ))
        step_num += 1
        
        # Get the degree of the polynomial
        degree = sp.degree(expr, gen=variable)
        
        if degree == 1:
            # Linear equation: ax + b = 0
            coeff = sp.diff(expr, variable)
            constant = expr.subs(variable, 0)
            
            steps.append(SolutionStep(
                step_num,
                "identify_terms",
                new_equation,
                new_equation,
                f"Identify the coefficient of {variable}: {sympy_to_latex(coeff)} and the constant term: {sympy_to_latex(constant)}"
            ))
            step_num += 1
            
            # Solve for x
            solution = -constant / coeff
            final_eq = sp.Eq(variable, solution)
            
            steps.append(SolutionStep(
                step_num,
                "solve",
                new_equation,
                final_eq,
                f"Solve for {variable} by dividing both sides by {sympy_to_latex(coeff)}: {sympy_to_latex(final_eq)}",
                hint="For a linear equation ax + b = 0, the solution is x = -b/a"
            ))
            
        elif degree == 2:
            # Quadratic equation: ax² + bx + c = 0
            a = sp.diff(expr, variable, 2) / 2
            b = sp.diff(expr, variable, 1) - a * 2 * variable
            b = b.subs(variable, 0)
            c = expr.subs(variable, 0)
            
            steps.append(SolutionStep(
                step_num,
                "identify_terms",
                new_equation,
                new_equation,
                f"Identify the coefficients: a = {sympy_to_latex(a)}, b = {sympy_to_latex(b)}, c = {sympy_to_latex(c)}"
            ))
            step_num += 1
            
            # Calculate discriminant
            discriminant = b**2 - 4*a*c
            
            steps.append(SolutionStep(
                step_num,
                "calculate_discriminant",
                new_equation,
                sp.Eq(sp.Symbol('Δ'), discriminant),
                f"Calculate the discriminant Δ = b² - 4ac = {sympy_to_latex(discriminant)}",
                hint="The discriminant determines the number and nature of solutions"
            ))
            step_num += 1
            
            # Apply quadratic formula
            if discriminant >= 0:
                x1 = (-b + sp.sqrt(discriminant)) / (2*a)
                x2 = (-b - sp.sqrt(discriminant)) / (2*a)
                
                steps.append(SolutionStep(
                    step_num,
                    "apply_quadratic_formula",
                    new_equation,
                    sp.Eq(variable, sp.Union(sp.FiniteSet(x1), sp.FiniteSet(x2))),
                    f"Apply the quadratic formula: {variable} = (-b ± √Δ)/2a to get {variable} = {sympy_to_latex(x1)} or {variable} = {sympy_to_latex(x2)}",
                    hint="For a quadratic equation ax² + bx + c = 0, the solutions are x = (-b ± √(b² - 4ac))/2a"
                ))
            else:
                steps.append(SolutionStep(
                    step_num,
                    "complex_solutions",
                    new_equation,
                    new_equation,
                    f"The discriminant is negative, so the equation has complex solutions",
                    hint="For a negative discriminant, the solutions are complex numbers"
                ))
        else:
            # Higher degree polynomial
            # For this implementation, we'll use SymPy's solve and show the factorization
            try:
                # Try to factor the polynomial
                factored = sp.factor(expr)
                
                if factored != expr:
                    steps.append(SolutionStep(
                        step_num,
                        "factorize",
                        new_equation,
                        sp.Eq(factored, 0),
                        f"Factorize the polynomial: {sympy_to_latex(sp.Eq(factored, 0))}",
                        hint="When a polynomial is factored, each factor set to zero gives a solution"
                    ))
                    step_num += 1
                
                # Get solutions
                solutions = sp.solve(expr, variable)
                
                solution_expr = sp.Eq(variable, sp.Union(*[sp.FiniteSet(sol) for sol in solutions]))
                
                steps.append(SolutionStep(
                    step_num,
                    "solve",
                    new_equation,
                    solution_expr,
                    f"The solutions are: {', '.join([sympy_to_latex(sp.Eq(variable, sol)) for sol in solutions])}",
                    hint=f"A polynomial of degree {degree} can have up to {degree} solutions"
                ))
            except Exception as e:
                steps.append(SolutionStep(
                    step_num,
                    "solve_complex",
                    new_equation,
                    new_equation,
                    f"This higher-degree polynomial requires more advanced techniques to solve: {e}",
                    hint="Higher-degree polynomials often don't have simple closed-form solutions"
                ))
        
        # Add verification step
        if step_num < len(steps):
            final_step = steps[-1]
            if hasattr(final_step, 'output_expr') and isinstance(final_step.output_expr, sp.Eq) and final_step.output_expr.lhs == variable:
                solutions = []
                if isinstance(final_step.output_expr.rhs, sp.Union):
                    for arg in final_step.output_expr.rhs.args:
                        if isinstance(arg, sp.FiniteSet) and len(arg.args) == 1:
                            solutions.append(arg.args[0])
                else:
                    solutions.append(final_step.output_expr.rhs)
                
                verification_explanations = []
                for solution in solutions:
                    result = equation.lhs.subs(variable, solution) - equation.rhs.subs(variable, solution)
                    verification_explanations.append(f"For {variable} = {sympy_to_latex(solution)}: {sympy_to_latex(equation.lhs)} = {sympy_to_latex(equation.lhs.subs(variable, solution))} and {sympy_to_latex(equation.rhs)} = {sympy_to_latex(equation.rhs.subs(variable, solution))}")
                
                steps.append(SolutionStep(
                    len(steps) + 1,
                    "verify",
                    equation,
                    final_step.output_expr,
                    "Verification: " + "; ".join(verification_explanations),
                    hint="Always verify your solutions by substituting back into the original equation"
                ))
        
        return steps
    
    def _solve_rational_equation(
        self,
        equation: sp.Eq,
        variables: List[sp.Symbol],
        steps: List[SolutionStep]
    ) -> List[SolutionStep]:
        """Handle rational equation solving with detailed steps."""
        step_num = len(steps) + 1
        variable = variables[0]  # Use the first variable
        
        # Find common denominator
        lhs, rhs = equation.lhs, equation.rhs
        
        # Try to get all terms to LHS
        expr = lhs - rhs
        new_equation = sp.Eq(expr, 0)
        
        steps.append(SolutionStep(
            step_num,
            "rearrange",
            equation,
            new_equation,
            f"Rearrange to get all terms on one side: {sympy_to_latex(new_equation)}"
        ))
        step_num += 1
        
        # Get the numerator and denominator
        num, den = expr.as_numer_denom()
        
        steps.append(SolutionStep(
            step_num,
            "identify_rational",
            new_equation,
            sp.Eq(sp.Mul(num, 1/den, evaluate=False), 0),
            f"Express as a fraction: {sympy_to_latex(num)}/{sympy_to_latex(den)} = 0",
            hint="A rational equation can be solved by setting the numerator to zero, after checking for domain restrictions"
        ))
        step_num += 1
        
        # Find excluded values
        try:
            domain_restrictions = sp.solve(den, variable)
            
            if domain_restrictions:
                restriction_text = ", ".join([f"{variable} ≠ {sympy_to_latex(val)}" for val in domain_restrictions])
                steps.append(SolutionStep(
                    step_num,
                    "domain_restrictions",
                    new_equation,
                    new_equation,
                    f"Note the domain restrictions: {restriction_text}",
                    hint="These values would cause division by zero and must be excluded from the solution set"
                ))
                step_num += 1
        except Exception as e:
            logger.warning(f"Could not determine domain restrictions: {e}")
        
        # Set numerator to zero and solve
        num_equation = sp.Eq(num, 0)
        
        steps.append(SolutionStep(
            step_num,
            "set_numerator_zero",
            new_equation,
            num_equation,
            f"For the rational expression to be zero, the numerator must be zero: {sympy_to_latex(num_equation)}",
            hint="When a rational expression equals zero, its numerator must be zero (if denominators are non-zero)"
        ))
        step_num += 1
        
        # Now solve the numerator equation (which might be polynomial)
        # We'll reuse our polynomial solver
        polynomial_steps = self._solve_polynomial_equation(num_equation, variables, [])
        
        # Add these steps, adjusting the step numbers
        for i, poly_step in enumerate(polynomial_steps, start=step_num):
            poly_step.step_number = i
            steps.append(poly_step)
        
        step_num += len(polynomial_steps)
        
        # Check if solutions satisfy domain restrictions
        if step_num > 1 and domain_restrictions:
            last_step = steps[-1]
            if hasattr(last_step, 'output_expr') and isinstance(last_step.output_expr, sp.Eq):
                if last_step.output_expr.lhs == variable:
                    solutions = []
                    if isinstance(last_step.output_expr.rhs, sp.Union):
                        for arg in last_step.output_expr.rhs.args:
                            if isinstance(arg, sp.FiniteSet) and len(arg.args) == 1:
                                solutions.append(arg.args[0])
                    else:
                        solutions.append(last_step.output_expr.rhs)
                    
                    # Check each solution against domain restrictions
                    valid_solutions = []
                    invalid_solutions = []
                    
                    for solution in solutions:
                        if solution in domain_restrictions:
                            invalid_solutions.append(solution)
                        else:
                            valid_solutions.append(solution)
                    
                    if invalid_solutions:
                        steps.append(SolutionStep(
                            step_num,
                            "check_domain",
                            last_step.output_expr,
                            last_step.output_expr,
                            f"The {'solutions' if len(invalid_solutions) > 1 else 'solution'} {', '.join(sympy_to_latex(sol) for sol in invalid_solutions)} {'are' if len(invalid_solutions) > 1 else 'is'} not valid because {'they violate' if len(invalid_solutions) > 1 else 'it violates'} the domain restrictions",
                            hint="Always check solutions against domain restrictions in rational equations"
                        ))
                        step_num += 1
                    
                    # If we filtered out any solutions, update the final answer
                    if invalid_solutions and valid_solutions:
                        if len(valid_solutions) == 1:
                            final_solution = sp.Eq(variable, valid_solutions[0])
                        else:
                            final_solution = sp.Eq(variable, sp.Union(*[sp.FiniteSet(sol) for sol in valid_solutions]))
                        
                        steps.append(SolutionStep(
                            step_num,
                            "final_solution",
                            last_step.output_expr,
                            final_solution,
                            f"The {'solutions are' if len(valid_solutions) > 1 else 'solution is'}: {', '.join(sympy_to_latex(sp.Eq(variable, sol)) for sol in valid_solutions)}",
                            hint="These are the solutions that satisfy both the equation and domain restrictions"
                        ))
        
        return steps
    
    def _solve_general_equation(
        self,
        equation: sp.Eq,
        variables: List[sp.Symbol],
        steps: List[SolutionStep]
    ) -> List[SolutionStep]:
        """Handle general equation solving with detailed steps."""
        step_num = len(steps) + 1
        variable = variables[0]  # Use the first variable
        
        # For general equations, we'll use a more flexible approach
        try:
            # Try to isolate the variable by manipulation
            # First, move all terms with the variable to the left, constants to the right
            lhs, rhs = equation.lhs, equation.rhs
            
            # Collect terms with the variable on LHS
            lhs_with_var = 0
            lhs_without_var = 0
            
            for term in lhs.as_ordered_terms():
                if variable in term.free_symbols:
                    lhs_with_var += term
                else:
                    lhs_without_var += term
            
            rhs_with_var = 0
            rhs_without_var = 0
            
            for term in rhs.as_ordered_terms():
                if variable in term.free_symbols:
                    rhs_with_var += term
                else:
                    rhs_without_var += term
            
            # Move variable terms to LHS, constants to RHS
            new_lhs = lhs_with_var - rhs_with_var
            new_rhs = rhs_without_var - lhs_without_var
            
            new_equation = sp.Eq(new_lhs, new_rhs)
            
            if new_equation != equation:
                steps.append(SolutionStep(
                    step_num,
                    "rearrange",
                    equation,
                    new_equation,
                    f"Rearrange the equation to isolate terms with {variable} on the left side: {sympy_to_latex(new_equation)}",
                    hint=f"Group all terms containing {variable} on one side of the equation"
                ))
                step_num += 1
            
            # Now try to factor out the variable if possible
            if isinstance(new_lhs, sp.Add):
                try:
                    factored = sp.factor(new_lhs)
                    if factored != new_lhs:
                        factored_eq = sp.Eq(factored, new_rhs)
                        steps.append(SolutionStep(
                            step_num,
                            "factor",
                            new_equation,
                            factored_eq,
                            f"Factor the left side: {sympy_to_latex(factored_eq)}",
                            hint="Factoring helps isolate the variable"
                        ))
                        step_num += 1
                        new_equation = factored_eq
                except Exception as e:
                    logger.warning(f"Factoring failed: {e}")
            
            # Try to solve using SymPy's solve function
            solutions = sp.solve(equation, variable)
            
            if solutions:
                if len(solutions) == 1:
                    final_solution = sp.Eq(variable, solutions[0])
                else:
                    final_solution = sp.Eq(variable, sp.Union(*[sp.FiniteSet(sol) for sol in solutions]))
                
                steps.append(SolutionStep(
                    step_num,
                    "solve",
                    new_equation,
                    final_solution,
                    f"Solve for {variable} to get: {', '.join([sympy_to_latex(sp.Eq(variable, sol)) for sol in solutions])}",
                    hint="This equation requires algebraic manipulation to isolate the variable"
                ))
                step_num += 1
                
                # Add verification step
                verification_explanations = []
                for solution in solutions:
                    lhs_eval = equation.lhs.subs(variable, solution)
                    rhs_eval = equation.rhs.subs(variable, solution)
                    verification_explanations.append(f"For {variable} = {sympy_to_latex(solution)}: {sympy_to_latex(equation.lhs)} = {sympy_to_latex(lhs_eval)} and {sympy_to_latex(equation.rhs)} = {sympy_to_latex(rhs_eval)}")
                
                steps.append(SolutionStep(
                    step_num,
                    "verify",
                    equation,
                    final_solution,
                    "Verification: " + "; ".join(verification_explanations),
                    hint="Always verify your solutions by substituting back into the original equation"
                ))
            else:
                steps.append(SolutionStep(
                    step_num,
                    "no_solution",
                    equation,
                    equation,
                    "This equation has no solution",
                    hint="Not all equations have solutions within the real numbers"
                ))
        except Exception as e:
            logger.error(f"General equation solving failed: {e}")
            steps.append(SolutionStep(
                step_num,
                "error",
                equation,
                equation,
                f"Could not solve this equation with the general approach: {e}",
                hint="This equation may require specialized techniques"
            ))
        
        return steps
    
    def _generate_differentiation_steps(
        self,
        expression: Union[sp.Expr, str],
        variables: Optional[List[Union[sp.Symbol, str]]] = None,
        domain: str = "calculus",
        context: Optional[Dict[str, Any]] = None
    ) -> List[SolutionStep]:
        """
        Generate steps for differentiation.
        
        Args:
            expression: The expression to differentiate
            variables: Variables to differentiate with respect to
            domain: Mathematical domain
            context: Additional context
            
        Returns:
            List of solution steps
        """
        steps = []
        context = context or {}
        
        # Convert string expression to sympy if needed
        if isinstance(expression, str):
            try:
                expression = sp.sympify(expression)
            except Exception as e:
                logger.error(f"Failed to parse expression: {e}")
                return [SolutionStep(
                    1, "error", expression, expression,
                    f"Error parsing expression: {e}"
                )]
        
        # Extract variable if not provided
        if not variables or len(variables) == 0:
            if hasattr(expression, 'free_symbols') and expression.free_symbols:
                variables = [list(expression.free_symbols)[0]]
            else:
                return [SolutionStep(
                    1, "error", expression, expression,
                    "No variables found in expression for differentiation"
                )]
        
        # Ensure variables are sympy Symbols
        var = variables[0]
        if isinstance(var, str):
            var = sp.Symbol(var)
        
        # Add initial step
        steps.append(SolutionStep(
            1,
            "initial",
            expression,
            expression,
            f"Find the derivative of {sympy_to_latex(expression)} with respect to {sympy_to_latex(var)}"
        ))
        
        # Check the type of expression to apply appropriate differentiation rules
        if isinstance(expression, sp.Add):
            # Sum rule: d/dx[f(x) + g(x)] = d/dx[f(x)] + d/dx[g(x)]
            step_num = 2
            terms = expression.args
            
            steps.append(SolutionStep(
                step_num,
                "apply_sum_rule",
                expression,
                expression,
                f"Apply the sum rule: The derivative of a sum is the sum of the derivatives",
                hint="Sum Rule: d/dx[f(x) + g(x)] = d/dx[f(x)] + d/dx[g(x)]"
            ))
            step_num += 1
            
            # Process each term
            result_terms = []
            for i, term in enumerate(terms):
                # Differentiate the term
                derivative = sp.diff(term, var)
                result_terms.append(derivative)
                
                steps.append(SolutionStep(
                    step_num,
                    "differentiate_term",
                    term,
                    derivative,
                    f"Differentiate term {i+1}: d/d{var}[{sympy_to_latex(term)}] = {sympy_to_latex(derivative)}",
                    hint=self._get_differentiation_rule_hint(term, var)
                ))
                step_num += 1
            
            # Combine results
            final_result = sum(result_terms)
            
            steps.append(SolutionStep(
                step_num,
                "combine_terms",
                expression,
                final_result,
                f"Combine all terms to get the final derivative: {sympy_to_latex(final_result)}",
                hint="The derivative is the sum of the derivatives of each term"
            ))
            
        elif isinstance(expression, sp.Mul):
            # Product rule or other rules depending on the factors
            factors = expression.args
            
            if len(factors) == 2:
                # Apply product rule: d/dx[f(x)g(x)] = f(x)·d/dx[g(x)] + g(x)·d/dx[f(x)]
                f, g = factors
                
                steps.append(SolutionStep(
                    2,
                    "identify_factors",
                    expression,
                    expression,
                    f"Identify the factors: {sympy_to_latex(expression)} = {sympy_to_latex(f)} · {sympy_to_latex(g)}",
                    hint="When differentiating a product, we'll use the product rule"
                ))
                
                steps.append(SolutionStep(
                    3,
                    "apply_product_rule",
                    expression,
                    expression,
                    f"Apply the product rule: d/d{var}[f·g] = f·dg/d{var} + g·df/d{var}",
                    hint="Product Rule: d/dx[f(x)·g(x)] = f(x)·d/dx[g(x)] + g(x)·d/dx[f(x)]"
                ))
                
                # Calculate derivatives of factors
                df = sp.diff(f, var)
                dg = sp.diff(g, var)
                
                steps.append(SolutionStep(
                    4,
                    "differentiate_first_factor",
                    f,
                    df,
                    f"Find the derivative of the first factor: d/d{var}[{sympy_to_latex(f)}] = {sympy_to_latex(df)}",
                    hint=self._get_differentiation_rule_hint(f, var)
                ))
                
                steps.append(SolutionStep(
                    5,
                    "differentiate_second_factor",
                    g,
                    dg,
                    f"Find the derivative of the second factor: d/d{var}[{sympy_to_latex(g)}] = {sympy_to_latex(dg)}",
                    hint=self._get_differentiation_rule_hint(g, var)
                ))
                
                # Apply product rule formula
                term1 = f * dg
                term2 = g * df
                result = term1 + term2
                
                steps.append(SolutionStep(
                    6,
                    "substitute_product_rule",
                    expression,
                    result,
                    f"Substitute into the product rule: {sympy_to_latex(f)}·{sympy_to_latex(dg)} + {sympy_to_latex(g)}·{sympy_to_latex(df)} = {sympy_to_latex(result)}",
                    hint="Substitute the derivatives into the product rule formula"
                ))
                
                # Simplify if possible
                simplified = sp.simplify(result)
                if simplified != result:
                    steps.append(SolutionStep(
                        7,
                        "simplify",
                        result,
                        simplified,
                        f"Simplify the result: {sympy_to_latex(simplified)}",
                        hint="Algebraic simplification can make the result clearer"
                    ))
            
            else:
                # For multiple factors, use recursive approach or general formula
                steps.append(SolutionStep(
                    2,
                    "multiple_factors",
                    expression,
                    expression,
                    f"This expression has multiple factors, requiring repeated application of the product rule",
                    hint="For multiple factors, we can apply the product rule recursively"
                ))
                
                # Use SymPy's differentiation directly for multiple factors
                result = sp.diff(expression, var)
                
                steps.append(SolutionStep(
                    3,
                    "final_result",
                    expression,
                    result,
                    f"The derivative is: {sympy_to_latex(result)}",
                    hint="This is obtained by repeatedly applying the product rule"
                ))
        
        elif expression.has(sp.sin) or expression.has(sp.cos) or expression.has(sp.tan) or \
             expression.has(sp.exp) or expression.has(sp.log):
            # Handle common functions with special rules
            steps.append(SolutionStep(
                2,
                "identify_function",
                expression,
                expression,
                f"Identify the type of function in the expression",
                hint="Different functions have different differentiation rules"
            ))
            
            # Use SymPy's diff directly
            result = sp.diff(expression, var)
            
            steps.append(SolutionStep(
                3,
                "apply_differentiation_rule",
                expression,
                result,
                f"Apply the appropriate differentiation rule to get: {sympy_to_latex(result)}",
                hint=self._get_differentiation_rule_hint(expression, var)
            ))
            
            # Check for chain rule application
            if expression.has(var) and not (expression.is_polynomial(var) or expression == var):
                steps[-1].hint = "This involves the chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x)"
        
        else:
            # General case - use SymPy's differentiation directly
            result = sp.diff(expression, var)
            
            steps.append(SolutionStep(
                2,
                "differentiate",
                expression,
                result,
                f"Apply the differentiation rules to get: {sympy_to_latex(result)}",
                hint="Use the appropriate differentiation rules based on the expression"
            ))
            
            # Simplify if possible
            simplified = sp.simplify(result)
            if simplified != result:
                steps.append(SolutionStep(
                    3,
                    "simplify",
                    result,
                    simplified,
                    f"Simplify the result: {sympy_to_latex(simplified)}",
                    hint="Algebraic simplification can make the result clearer"
                ))
        
        return steps
    
    def _get_differentiation_rule_hint(self, expression: sp.Expr, var: sp.Symbol) -> str:
        """Get a hint about the differentiation rule applied to an expression."""
        # Constant rule
        if not expression.has(var):
            return "Constant Rule: d/dx[c] = 0"
        
        # Power rule
        if expression.is_Pow and expression.args[0] == var:
            return f"Power Rule: d/dx[x^n] = n·x^(n-1)"
        
        # Exponential
        if expression.has(sp.exp):
            return "Exponential Rule: d/dx[e^x] = e^x"
        
        # Trigonometric functions
        if expression.has(sp.sin):
            return "Sine Rule: d/dx[sin(x)] = cos(x)"
        if expression.has(sp.cos):
            return "Cosine Rule: d/dx[cos(x)] = -sin(x)"
        if expression.has(sp.tan):
            return "Tangent Rule: d/dx[tan(x)] = sec²(x)"
        
        # Logarithmic
        if expression.has(sp.log):
            return "Logarithm Rule: d/dx[ln(x)] = 1/x"
        
        # Chain rule (general case)
        if expression.has(var) and not expression.is_polynomial(var):
            return "Chain Rule: d/dx[f(g(x))] = f'(g(x))·g'(x)"
        
        return "Apply the appropriate differentiation rule"
    
    def _generate_integration_steps(
        self,
        expression: Union[sp.Expr, str],
        variables: Optional[List[Union[sp.Symbol, str]]] = None,
        domain: str = "calculus",
        context: Optional[Dict[str, Any]] = None
    ) -> List[SolutionStep]:
        """
        Generate steps for integration.
        
        Args:
            expression: The expression to integrate
            variables: Variables to integrate with respect to
            domain: Mathematical domain
            context: Additional context
            
        Returns:
            List of solution steps
        """
        steps = []
        context = context or {}
        
        # Convert string expression to sympy if needed
        if isinstance(expression, str):
            try:
                expression = sp.sympify(expression)
            except Exception as e:
                logger.error(f"Failed to parse expression: {e}")
                return [SolutionStep(
                    1, "error", expression, expression,
                    f"Error parsing expression: {e}"
                )]
        
        # Extract variable if not provided
        if not variables or len(variables) == 0:
            if hasattr(expression, 'free_symbols') and expression.free_symbols:
                variables = [list(expression.free_symbols)[0]]
            else:
                return [SolutionStep(
                    1, "error", expression, expression,
                    "No variables found in expression for integration"
                )]
        
        # Ensure variables are sympy Symbols
        var = variables[0]
        if isinstance(var, str):
            var = sp.Symbol(var)
        
        # Check for definite integration
        limits = None
        if context and 'limits' in context:
            limits = context['limits']
            
            # Add initial step for definite integration
            limits_str = f" from {sympy_to_latex(limits[0])} to {sympy_to_latex(limits[1])}"
            steps.append(SolutionStep(
                1,
                "initial",
                expression,
                expression,
                f"Find the definite integral of {sympy_to_latex(expression)} with respect to {sympy_to_latex(var)}{limits_str}"
            ))
        else:
            # Add initial step for indefinite integration
            steps.append(SolutionStep(
                1,
                "initial",
                expression,
                expression,
                f"Find the indefinite integral of {sympy_to_latex(expression)} with respect to {sympy_to_latex(var)}"
            ))
        
        # Check the type of expression to apply appropriate integration rules
        if isinstance(expression, sp.Add):
            # Sum rule: ∫(f(x) + g(x))dx = ∫f(x)dx + ∫g(x)dx
            step_num = 2
            terms = expression.args
            
            steps.append(SolutionStep(
                step_num,
                "apply_sum_rule",
                expression,
                expression,
                f"Apply the sum rule: The integral of a sum is the sum of the integrals",
                hint="Sum Rule: ∫[f(x) + g(x)]dx = ∫f(x)dx + ∫g(x)dx"
            ))
            step_num += 1
            
            # Process each term
            result_terms = []
            for i, term in enumerate(terms):
                # Integrate the term
                integral = sp.integrate(term, var)
                result_terms.append(integral)
                
                steps.append(SolutionStep(
                    step_num,
                    "integrate_term",
                    term,
                    integral,
                    f"Integrate term {i+1}: ∫{sympy_to_latex(term)}d{var} = {sympy_to_latex(integral)} + C",
                    hint=self._get_integration_rule_hint(term, var)
                ))
                step_num += 1
            
            # Combine results
            antiderivative = sum(result_terms)
            
            if limits:
                # For definite integration, evaluate at the limits
                upper_eval = antiderivative.subs(var, limits[1])
                lower_eval = antiderivative.subs(var, limits[0])
                result = upper_eval - lower_eval
                
                steps.append(SolutionStep(
                    step_num,
                    "evaluate_at_limits",
                    antiderivative,
                    result,
                    f"Evaluate the antiderivative at the limits: {sympy_to_latex(antiderivative)}|_{limits[0]}^{limits[1]} = {sympy_to_latex(upper_eval)} - {sympy_to_latex(lower_eval)} = {sympy_to_latex(result)}",
                    hint="For definite integrals, evaluate the antiderivative at the upper and lower limits"
                ))
            else:
                # For indefinite integration, add the constant of integration
                steps.append(SolutionStep(
                    step_num,
                    "combine_terms",
                    expression,
                    antiderivative,
                    f"Combine all terms to get the antiderivative: {sympy_to_latex(antiderivative)} + C",
                    hint="Don't forget to add the constant of integration"
                ))
                
        else:
            # For other types of expressions, apply specific integration techniques
            
            # Try to identify the type of integral
            if expression.is_polynomial(var):
                steps.append(SolutionStep(
                    2,
                    "identify_polynomial",
                    expression,
                    expression,
                    f"This is a polynomial in {var}, we can integrate term by term",
                    hint=f"For polynomials, integrate each term using the power rule: ∫x^n dx = x^(n+1)/(n+1) + C for n ≠ -1"
                ))
                
                antiderivative = sp.integrate(expression, var)
                
                steps.append(SolutionStep(
                    3,
                    "apply_power_rule",
                    expression,
                    antiderivative,
                    f"Apply the power rule to get: {sympy_to_latex(antiderivative)} + C",
                    hint="Power Rule: ∫x^n dx = x^(n+1)/(n+1) + C for n ≠ -1"
                ))
                
            elif expression.has(sp.sin) or expression.has(sp.cos):
                steps.append(SolutionStep(
                    2,
                    "identify_trig",
                    expression,
                    expression,
                    f"This expression contains trigonometric functions",
                    hint="Use trigonometric integration rules"
                ))
                
                antiderivative = sp.integrate(expression, var)
                
                steps.append(SolutionStep(
                    3,
                    "apply_trig_rule",
                    expression,
                    antiderivative,
                    f"Apply the appropriate trigonometric integration rule to get: {sympy_to_latex(antiderivative)} + C",
                    hint="Trigonometric Rules: ∫sin(x)dx = -cos(x) + C, ∫cos(x)dx = sin(x) + C"
                ))
                
            elif (expression.is_Pow and expression.args[0] == var and expression.args[1] == -1) or \
                 expression == 1/var:
                steps.append(SolutionStep(
                    2,
                    "identify_log",
                    expression,
                    expression,
                    f"This is of the form 1/{var}, which integrates to the natural logarithm",
                    hint="For 1/x, the integral is ln|x| + C"
                ))
                
                antiderivative = sp.integrate(expression, var)
                
                steps.append(SolutionStep(
                    3,
                    "apply_log_rule",
                    expression,
                    antiderivative,
                    f"Apply the logarithm rule to get: {sympy_to_latex(antiderivative)} + C",
                    hint="Logarithm Rule: ∫(1/x)dx = ln|x| + C"
                ))
                
            elif expression.has(sp.exp(var)):
                steps.append(SolutionStep(
                    2,
                    "identify_exp",
                    expression,
                    expression,
                    f"This expression contains e^{var}",
                    hint="The integral of e^x is e^x + C"
                ))
                
                antiderivative = sp.integrate(expression, var)
                
                steps.append(SolutionStep(
                    3,
                    "apply_exp_rule",
                    expression,
                    antiderivative,
                    f"Apply the exponential rule to get: {sympy_to_latex(antiderivative)} + C",
                    hint="Exponential Rule: ∫e^x dx = e^x + C"
                ))
                
            else:
                # General case - use SymPy's integration directly
                steps.append(SolutionStep(
                    2,
                    "general_integration",
                    expression,
                    expression,
                    f"For this expression, we need to apply appropriate integration techniques",
                    hint="Integration may require substitution, parts, or other methods"
                ))
                
                antiderivative = sp.integrate(expression, var)
                
                steps.append(SolutionStep(
                    3,
                    "integrate",
                    expression,
                    antiderivative,
                    f"Applying integration rules: {sympy_to_latex(antiderivative)} + C",
                    hint="The antiderivative is found using appropriate integration techniques"
                ))
            
            # For definite integration, evaluate at limits
            if limits:
                step_num = len(steps) + 1
                
                # Evaluate at the limits
                upper_eval = antiderivative.subs(var, limits[1])
                lower_eval = antiderivative.subs(var, limits[0])
                result = upper_eval - lower_eval
                
                steps.append(SolutionStep(
                    step_num,
                    "evaluate_at_limits",
                    antiderivative,
                    result,
                    f"Evaluate the antiderivative at the limits: {sympy_to_latex(antiderivative)}|_{limits[0]}^{limits[1]} = {sympy_to_latex(upper_eval)} - {sympy_to_latex(lower_eval)} = {sympy_to_latex(result)}",
                    hint="For definite integrals, evaluate the antiderivative at the upper and lower limits"
                ))
        
        return steps
    
    def _get_integration_rule_hint(self, expression: sp.Expr, var: sp.Symbol) -> str:
        """Get a hint about the integration rule applied to an expression."""
        # Constant rule
        if not expression.has(var):
            return "Constant Rule: ∫c dx = c·x + C"
        
        # Power rule
        if expression.is_Pow and expression.args[0] == var:
            n = expression.args[1]
            if n == -1:
                return "Logarithm Rule: ∫(1/x)dx = ln|x| + C"
            else:
                return f"Power Rule: ∫x^n dx = x^(n+1)/(n+1) + C for n ≠ -1"
        
        # Exponential
        if expression.has(sp.exp):
            return "Exponential Rule: ∫e^x dx = e^x + C"
        
        # Trigonometric functions
        if expression.has(sp.sin):
            return "Sine Rule: ∫sin(x)dx = -cos(x) + C"
        if expression.has(sp.cos):
            return "Cosine Rule: ∫cos(x)dx = sin(x) + C"
        if expression.has(sp.tan):
            return "Tangent Rule: ∫tan(x)dx = -ln|cos(x)| + C"
        
        # General case
        return "Apply the appropriate integration rule"
    
    def _generate_limit_steps(
        self,
        expression: Union[sp.Expr, str],
        variables: Optional[List[Union[sp.Symbol, str]]] = None,
        domain: str = "calculus",
        context: Optional[Dict[str, Any]] = None
    ) -> List[SolutionStep]:
        """
        Generate steps for evaluating limits.
        
        Args:
            expression: The expression for the limit
            variables: Variable to evaluate the limit for
            domain: Mathematical domain
            context: Additional context including the limit point
            
        Returns:
            List of solution steps
        """
        steps = []
        context = context or {}
        
        # Extract point where limit is evaluated
        if 'limit_point' not in context:
            return [SolutionStep(
                1, "error", expression, expression,
                "No limit point specified for limit evaluation"
            )]
        
        limit_point = context['limit_point']
        approach_dir = context.get('approach_dir', None)  # Direction of approach ('+' or '-' or None)
        
        # Convert string expression to sympy if needed
        if isinstance(expression, str):
            try:
                expression = sp.sympify(expression)
            except Exception as e:
                logger.error(f"Failed to parse expression: {e}")
                return [SolutionStep(
                    1, "error", expression, expression,
                    f"Error parsing expression: {e}"
                )]
        
        # Extract variable if not provided
        if not variables or len(variables) == 0:
            if hasattr(expression, 'free_symbols') and expression.free_symbols:
                variables = [list(expression.free_symbols)[0]]
            else:
                return [SolutionStep(
                    1, "error", expression, expression,
                    "No variables found in expression for limit"
                )]
        
        # Ensure variables are sympy Symbols
        var = variables[0]
        if isinstance(var, str):
            var = sp.Symbol(var)
        
        # Add initial step
        approach_str = ""
        if approach_dir == '+':
            approach_str = f" from the right"
        elif approach_dir == '-':
            approach_str = f" from the left"
            
        steps.append(SolutionStep(
            1,
            "initial",
            expression,
            expression,
            f"Evaluate the limit of {sympy_to_latex(expression)} as {sympy_to_latex(var)} approaches {sympy_to_latex(limit_point)}{approach_str}"
        ))
        
        # Try direct substitution first
        try:
            direct_subst = expression.subs(var, limit_point)
            
            # Check if substitution gives a determinable result
            if not (sp.oo in direct_subst.atoms() or -sp.oo in direct_subst.atoms() or
                   sp.zoo in direct_subst.atoms() or direct_subst.has(sp.nan)):
                
                steps.append(SolutionStep(
                    2,
                    "direct_substitution",
                    expression,
                    direct_subst,
                    f"Try direct substitution: {sympy_to_latex(expression)} with {var} = {sympy_to_latex(limit_point)} gives {sympy_to_latex(direct_subst)}",
                    hint="If direct substitution gives a determinate value, that is the limit"
                ))
                
                # Verify the limit with SymPy's limit function
                if approach_dir:
                    limit_result = sp.limit(expression, var, limit_point, dir=approach_dir)
                else:
                    limit_result = sp.limit(expression, var, limit_point)
                
                if limit_result == direct_subst:
                    steps.append(SolutionStep(
                        3,
                        "confirm_result",
                        direct_subst,
                        limit_result,
                        f"The limit is {sympy_to_latex(limit_result)}",
                        hint="The limit was found by direct substitution"
                    ))
                    return steps
                
                # If direct substitution doesn't match the actual limit, continue with more techniques
                steps.append(SolutionStep(
                    3,
                    "substitution_incorrect",
                    direct_subst,
                    direct_subst,
                    f"However, direct substitution gives an incorrect result. We need to use other techniques",
                    hint="Direct substitution doesn't always give the correct limit"
                ))
            else:
                # If substitution gives an indeterminate form
                steps.append(SolutionStep(
                    2,
                    "indeterminate_form",
                    expression,
                    direct_subst,
                    f"Direct substitution yields an indeterminate form: {sympy_to_latex(direct_subst)}",
                    hint="When substitution gives an indeterminate form, we need to use algebraic manipulation or L'Hôpital's rule"
                ))
        except Exception as e:
            # If substitution fails
            steps.append(SolutionStep(
                2,
                "substitution_error",
                expression,
                expression,
                f"Direct substitution fails: {e}",
                hint="When direct substitution fails, we need to use alternative techniques"
            ))
        
        # Check if this is a rational function that can be simplified
        try:
            num, den = expression.as_numer_denom()
            
            if den.subs(var, limit_point) == 0:
                # We have a division by zero situation
                steps.append(SolutionStep(
                    len(steps) + 1,
                    "rational_function",
                    expression,
                    expression,
                    f"This is a rational function with denominator approaching zero",
                    hint="For rational functions, try factoring and canceling common terms"
                ))
                
                # Try algebraic manipulation - factor numerator and denominator
                num_factored = sp.factor(num)
                den_factored = sp.factor(den)
                
                if num_factored != num or den_factored != den:
                    factored_expr = sp.Mul(num_factored, 1/den_factored, evaluate=False)
                    
                    steps.append(SolutionStep(
                        len(steps) + 1,
                        "factor",
                        expression,
                        factored_expr,
                        f"Factor the expression: {sympy_to_latex(factored_expr)}",
                        hint="Factoring can reveal common terms that can be canceled"
                    ))
                    
                    # Look for common factors that can be canceled
                    # This is a simplified approach - in practice, we'd need more sophisticated analysis
                    simplified = sp.simplify(expression)
                    if simplified != expression:
                        steps.append(SolutionStep(
                            len(steps) + 1,
                            "simplify",
                            factored_expr,
                            simplified,
                            f"Cancel common factors to get: {sympy_to_latex(simplified)}",
                            hint="After canceling common factors, direct substitution may work"
                        ))
                        
                        # Try substitution again on the simplified expression
                        try:
                            simplified_subst = simplified.subs(var, limit_point)
                            if not (sp.oo in simplified_subst.atoms() or -sp.oo in simplified_subst.atoms() or
                                   sp.zoo in simplified_subst.atoms() or simplified_subst.has(sp.nan)):
                                
                                steps.append(SolutionStep(
                                    len(steps) + 1,
                                    "substitute_simplified",
                                    simplified,
                                    simplified_subst,
                                    f"Substitute {var} = {sympy_to_latex(limit_point)} into the simplified expression: {sympy_to_latex(simplified_subst)}",
                                    hint="After simplification, substitution may give the limit"
                                ))
                                
                                # Final result
                                steps.append(SolutionStep(
                                    len(steps) + 1,
                                    "final_result",
                                    simplified_subst,
                                    simplified_subst,
                                    f"The limit is {sympy_to_latex(simplified_subst)}",
                                    hint="The limit was found by simplifying and then substituting"
                                ))
                                
                                return steps
                        except Exception:
                            pass
        except Exception:
            pass
        
        # If all else failed, use SymPy's limit function
        try:
            if approach_dir:
                limit_result = sp.limit(expression, var, limit_point, dir=approach_dir)
            else:
                limit_result = sp.limit(expression, var, limit_point)
            
            steps.append(SolutionStep(
                len(steps) + 1,
                "apply_advanced_techniques",
                expression,
                limit_result,
                f"Apply advanced limit techniques to get: {sympy_to_latex(limit_result)}",
                hint="Advanced techniques may include L'Hôpital's rule, algebraic manipulation, or special limits"
            ))
            
            # Describe the result
            if limit_result == sp.oo:
                steps.append(SolutionStep(
                    len(steps) + 1,
                    "infinite_limit",
                    limit_result,
                    limit_result,
                    f"The limit is positive infinity",
                    hint="The function grows without bound as x approaches the limit point"
                ))
            elif limit_result == -sp.oo:
                steps.append(SolutionStep(
                    len(steps) + 1,
                    "negative_infinite_limit",
                    limit_result,
                    limit_result,
                    f"The limit is negative infinity",
                    hint="The function decreases without bound as x approaches the limit point"
                ))
            elif limit_result == sp.zoo or limit_result.has(sp.nan):
                steps.append(SolutionStep(
                    len(steps) + 1,
                    "limit_dne",
                    limit_result,
                    limit_result,
                    f"The limit does not exist",
                    hint="The function's behavior is undefined at the limit point"
                ))
            else:
                steps.append(SolutionStep(
                    len(steps) + 1,
                    "final_result",
                    limit_result,
                    limit_result,
                    f"The limit is {sympy_to_latex(limit_result)}",
                    hint="The limit was found using advanced techniques"
                ))
        except Exception as e:
            steps.append(SolutionStep(
                len(steps) + 1,
                "limit_error",
                expression,
                expression,
                f"Could not determine the limit: {e}",
                hint="This limit may require specialized techniques or exist only conditionally"
            ))
        
        return steps
    
    def _generate_matrix_operation_steps(
        self,
        expression: Union[sp.Matrix, str],
        variables: Optional[List[Union[sp.Symbol, str]]] = None,
        domain: str = "linear_algebra",
        context: Optional[Dict[str, Any]] = None
    ) -> List[SolutionStep]:
        """
        Generate steps for matrix operations.
        
        Args:
            expression: The matrix or matrix expression
            variables: Variables in the expression
            domain: Mathematical domain
            context: Additional context including the operation type
            
        Returns:
            List of solution steps
        """
        steps = []
        context = context or {}
        
        # Get the operation type from context
        if 'operation_type' not in context:
            return [SolutionStep(
                1, "error", expression, expression,
                "No matrix operation type specified"
            )]
        
        operation_type = context['operation_type']
        
        # Convert string expression to sympy if needed
        if isinstance(expression, str):
            try:
                expression = sp.sympify(expression)
            except Exception as e:
                logger.error(f"Failed to parse expression: {e}")
                return [SolutionStep(
                    1, "error", expression, expression,
                    f"Error parsing matrix expression: {e}"
                )]
        
        # Add initial step
        steps.append(SolutionStep(
            1,
            "initial",
            expression,
            expression,
            f"Perform {operation_type} on the given matrix expression"
        ))
        
        # Handle different matrix operations
        if operation_type == "determinant":
            if not (hasattr(expression, 'is_Matrix') and expression.is_Matrix):
                return [SolutionStep(
                    2, "error", expression, expression,
                    "Cannot calculate determinant: Expression is not a matrix"
                )]
                
            # Check if it's a square matrix
            if expression.rows != expression.cols:
                return [SolutionStep(
                    2, "error", expression, expression,
                    f"Cannot calculate determinant: Matrix is not square (dimensions: {expression.rows}x{expression.cols})"
                )]
                
            # For 2x2 matrix, show the formula
            if expression.rows == 2:
                steps.append(SolutionStep(
                    2,
                    "determinant_formula_2x2",
                    expression,
                    expression,
                    f"For a 2×2 matrix, the determinant is calculated as: det(A) = a11·a22 - a12·a21",
                    hint="The determinant of a 2×2 matrix is the product of the main diagonal minus the product of the other diagonal"
                ))
                
                # Extract elements
                a11 = expression[0, 0]
                a12 = expression[0, 1]
                a21 = expression[1, 0]
                a22 = expression[1, 1]
                
                # Calculate determinant
                main_diag = a11 * a22
                other_diag = a12 * a21
                det = main_diag - other_diag
                
                steps.append(SolutionStep(
                    3,
                    "calculate_determinant_2x2",
                    expression,
                    det,
                    f"Calculate: det(A) = {sympy_to_latex(a11)}·{sympy_to_latex(a22)} - {sympy_to_latex(a12)}·{sympy_to_latex(a21)} = {sympy_to_latex(main_diag)} - {sympy_to_latex(other_diag)} = {sympy_to_latex(det)}",
                    hint="Substitute the matrix elements into the formula"
                ))
                
            # For 3x3 matrix, show the formula or cofactor expansion
            elif expression.rows == 3:
                steps.append(SolutionStep(
                    2,
                    "determinant_method_3x3",
                    expression,
                    expression,
                    f"For a 3×3 matrix, we can use the cofactor expansion method",
                    hint="The determinant can be calculated by expanding along any row or column"
                ))
                
                # Let's expand along the first row for simplicity
                steps.append(SolutionStep(
                    3,
                    "choose_expansion",
                    expression,
                    expression,
                    f"We'll expand along the first row",
                    hint="Expanding along a row with zeros (if any) can simplify calculations"
                ))
                
                # Extract first row elements
                a11 = expression[0, 0]
                a12 = expression[0, 1]
                a13 = expression[0, 2]
                
                # Calculate minors
                minor11 = expression.minor_submatrix(0, 0)
                minor12 = expression.minor_submatrix(0, 1)
                minor13 = expression.minor_submatrix(0, 2)
                
                steps.append(SolutionStep(
                    4,
                    "calculate_minors",
                    expression,
                    expression,
                    f"Calculate the minors for the first row elements",
                    hint="The minor of an element is the determinant of the submatrix formed by removing its row and column"
                ))
                
                # Calculate determinants of minors
                det11 = minor11.det()
                det12 = minor12.det()
                det13 = minor13.det()
                
                steps.append(SolutionStep(
                    5,
                    "calculate_cofactors",
                    expression,
                    expression,
                    f"Calculate the cofactors: C11 = {sympy_to_latex(det11)}, C12 = -{sympy_to_latex(det12)}, C13 = {sympy_to_latex(det13)}",
                    hint="Cofactor includes the sign, which alternates in a checkerboard pattern"
                ))
                
                # Calculate determinant using cofactor expansion
                term1 = a11 * det11
                term2 = -a12 * det12
                term3 = a13 * det13
                det = term1 + term2 + term3
                
                steps.append(SolutionStep(
                    6,
                    "apply_cofactor_expansion",
                    expression,
                    det,
                    f"Apply the cofactor expansion: det(A) = {sympy_to_latex(a11)}·{sympy_to_latex(det11)} + ({sympy_to_latex(-a12)})·{sympy_to_latex(det12)} + {sympy_to_latex(a13)}·{sympy_to_latex(det13)} = {sympy_to_latex(term1)} + {sympy_to_latex(term2)} + {sympy_to_latex(term3)} = {sympy_to_latex(det)}",
                    hint="The determinant is the sum of elements multiplied by their cofactors"
                ))
                
            # For larger matrices, use the determinant property
            else:
                steps.append(SolutionStep(
                    2,
                    "determinant_larger_matrix",
                    expression,
                    expression,
                    f"For larger matrices, we use computational methods for efficiency",
                    hint="Determinants of larger matrices can be calculated using row operations or cofactor expansion"
                ))
                
                # Calculate determinant using SymPy
                det = expression.det()
                
                steps.append(SolutionStep(
                    3,
                    "calculate_determinant",
                    expression,
                    det,
                    f"The determinant is: {sympy_to_latex(det)}",
                    hint="For practical purposes, computational methods are used for larger matrices"
                ))
                
        elif operation_type == "inverse":
            if not (hasattr(expression, 'is_Matrix') and expression.is_Matrix):
                return [SolutionStep(
                    2, "error", expression, expression,
                    "Cannot calculate inverse: Expression is not a matrix"
                )]
                
            # Check if it's a square matrix
            if expression.rows != expression.cols:
                return [SolutionStep(
                    2, "error", expression, expression,
                    f"Cannot calculate inverse: Matrix is not square (dimensions: {expression.rows}x{expression.cols})"
                )]
                
            # Calculate determinant to check invertibility
            det = expression.det()
            steps.append(SolutionStep(
                2,
                "check_invertibility",
                expression,
                det,
                f"Check if the matrix is invertible by calculating its determinant: det(A) = {sympy_to_latex(det)}",
                hint="A matrix is invertible if and only if its determinant is non-zero"
            ))
            
            if det == 0:
                steps.append(SolutionStep(
                    3,
                    "not_invertible",
                    expression,
                    expression,
                    f"The matrix is not invertible (singular) because its determinant is zero",
                    hint="Singular matrices do not have inverses"
                ))
                return steps
            
            # For 2x2 matrix, show the formula
            if expression.rows == 2:
                steps.append(SolutionStep(
                    3,
                    "inverse_formula_2x2",
                    expression,
                    expression,
                    f"For a 2×2 matrix, the inverse is calculated using the formula: A^(-1) = (1/det(A)) · [d -b; -c a]",
                    hint="The inverse of a 2×2 matrix involves swapping the diagonal elements, negating the off-diagonal elements, and dividing by the determinant"
                ))
                
                # Extract elements
                a = expression[0, 0]
                b = expression[0, 1]
                c = expression[1, 0]
                d = expression[1, 1]
                
                # Adjugate matrix
                adj = sp.Matrix([[d, -b], [-c, a]])
                
                steps.append(SolutionStep(
                    4,
                    "calculate_adjugate_2x2",
                    expression,
                    adj,
                    f"Create the adjugate matrix by swapping diagonal elements and negating off-diagonal elements: adj(A) = [{sympy_to_latex(d)}, {sympy_to_latex(-b)}; {sympy_to_latex(-c)}, {sympy_to_latex(a)}]",
                    hint="The adjugate matrix is formed from the cofactors"
                ))
                
                # Calculate inverse
                inverse = adj / det
                
                steps.append(SolutionStep(
                    5,
                    "calculate_inverse_2x2",
                    adj,
                    inverse,
                    f"Divide the adjugate matrix by the determinant: A^(-1) = (1/{sympy_to_latex(det)}) · [{sympy_to_latex(d)}, {sympy_to_latex(-b)}; {sympy_to_latex(-c)}, {sympy_to_latex(a)}] = {sympy_to_latex(inverse)}",
                    hint="The inverse is the adjugate divided by the determinant"
                ))
                
            # For larger matrices, use the adjugate method or Gauss-Jordan elimination
            else:
                steps.append(SolutionStep(
                    3,
                    "inverse_larger_matrix",
                    expression,
                    expression,
                    f"For larger matrices, we commonly use the Gauss-Jordan elimination method",
                    hint="The inverse can be found by row-reducing [A|I] to [I|A^(-1)]"
                ))
                
                # Calculate inverse using SymPy
                inverse = expression.inv()
                
                steps.append(SolutionStep(
                    4,
                    "calculate_inverse",
                    expression,
                    inverse,
                    f"The inverse matrix is: {sympy_to_latex(inverse)}",
                    hint="For practical purposes, computational methods are used for larger matrices"
                ))
                
        elif operation_type == "eigenvalues":
            if not (hasattr(expression, 'is_Matrix') and expression.is_Matrix):
                return [SolutionStep(
                    2, "error", expression, expression,
                    "Cannot calculate eigenvalues: Expression is not a matrix"
                )]
                
            # Check if it's a square matrix
            if expression.rows != expression.cols:
                return [SolutionStep(
                    2, "error", expression, expression,
                    f"Cannot calculate eigenvalues: Matrix is not square (dimensions: {expression.rows}x{expression.cols})"
                )]
                
            steps.append(SolutionStep(
                2,
                "eigenvalue_definition",
                expression,
                expression,
                f"Eigenvalues are values λ such that Ax = λx for some non-zero vector x",
                hint="Eigenvalues are found by solving the characteristic equation det(A - λI) = 0"
            ))
            
            # Define the eigenvalue symbol
            lambda_sym = sp.Symbol('λ')
            
            # Create the characteristic matrix
            char_matrix = expression - lambda_sym * sp.eye(expression.rows)
            
            steps.append(SolutionStep(
                3,
                "characteristic_matrix",
                expression,
                char_matrix,
                f"Form the characteristic matrix: A - λI = {sympy_to_latex(char_matrix)}",
                hint="Subtract λ from each diagonal element of the matrix"
            ))
            
            # Calculate the characteristic polynomial
            char_poly = char_matrix.det()
            
            steps.append(SolutionStep(
                4,
                "characteristic_polynomial",
                char_matrix,
                char_poly,
                f"Calculate the characteristic polynomial: det(A - λI) = {sympy_to_latex(char_poly)}",
                hint="The characteristic polynomial is the determinant of the characteristic matrix"
            ))
            
            # Solve for eigenvalues
            eigenvalues = sp.solve(char_poly, lambda_sym)
            
            steps.append(SolutionStep(
                5,
                "solve_eigenvalues",
                char_poly,
                eigenvalues,
                f"Solve the characteristic equation det(A - λI) = 0 to find the eigenvalues: {', '.join([sympy_to_latex(val) for val in eigenvalues])}",
                hint="The eigenvalues are the roots of the characteristic polynomial"
            ))
            
        else:
            # For other operations, provide a general approach
            steps.append(SolutionStep(
                2,
                "general_operation",
                expression,
                expression,
                f"For {operation_type} operation, we need to apply appropriate matrix techniques",
                hint=f"Matrix {operation_type} follows specific rules in linear algebra"
            ))
            
            # Use SymPy to calculate the result
            result = None
            try:
                if operation_type == "transpose":
                    result = expression.transpose()
                    steps.append(SolutionStep(
                        3,
                        "calculate_transpose",
                        expression,
                        result,
                        f"The transpose is: {sympy_to_latex(result)}",
                        hint="The transpose of a matrix flips its rows and columns"
                    ))
                elif operation_type == "rank":
                    result = expression.rank()
                    steps.append(SolutionStep(
                        3,
                        "calculate_rank",
                        expression,
                        result,
                        f"The rank is: {result}",
                        hint="The rank is the dimension of the vector space spanned by the matrix's columns"
                    ))
                else:
                    steps.append(SolutionStep(
                        3,
                        "operation_not_implemented",
                        expression,
                        expression,
                        f"The operation {operation_type} is not specifically implemented in this step generator",
                        hint="You may need to consult linear algebra resources for this operation"
                    ))
            except Exception as e:
                steps.append(SolutionStep(
                    3,
                    "operation_error",
                    expression,
                    expression,
                    f"Could not perform {operation_type}: {e}",
                    hint="This operation may require additional conditions or preprocessing"
                ))
        
        return steps

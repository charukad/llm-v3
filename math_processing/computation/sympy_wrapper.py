"""
SymPy Wrapper - provides symbolic mathematical computation capabilities.

This module wraps SymPy functionality to provide symbolic mathematics operations
including algebra, calculus, and linear algebra.
"""

import sympy as sp
from typing import Dict, List, Optional, Union, Any, Tuple


class SymbolicProcessor:
    """Wrapper for SymPy providing symbolic mathematical operations."""
    
    def __init__(self):
        """Initialize the symbolic processor."""
        # Common symbols that might be used in expressions
        self._initialize_common_symbols()
    
    def _initialize_common_symbols(self):
        """Initialize common symbols used in mathematical expressions."""
        # Create common variable symbols
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.z = sp.Symbol('z')
        self.t = sp.Symbol('t')
        
        # Common symbols dictionary for quick lookup
        self.symbols = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            't': self.t
        }
    
    def get_symbol(self, name: str) -> sp.Symbol:
        """
        Get a symbol by name, creating it if it doesn't exist.
        
        Args:
            name: Symbol name
            
        Returns:
            SymPy symbol
        """
        if name in self.symbols:
            return self.symbols[name]
        else:
            symbol = sp.Symbol(name)
            self.symbols[name] = symbol
            return symbol
    
    def solve_equation(self, 
                      equation: Union[sp.Eq, str], 
                      variable: Union[sp.Symbol, str, None] = None) -> Any:
        """
        Solve an equation for a variable.
        
        Args:
            equation: SymPy equation or string representation
            variable: Variable to solve for (if None, solver will try to determine)
            
        Returns:
            Solution(s) to the equation
        """
        # Handle string input
        if isinstance(equation, str):
            if '=' in equation:
                lhs, rhs = equation.split('=', 1)
                equation = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            else:
                # If no equals sign, assume equation = 0
                equation = sp.Eq(sp.sympify(equation), 0)
        
        # Handle non-equation input (assume equals 0)
        if not isinstance(equation, sp.Eq):
            equation = sp.Eq(equation, 0)
        
        # Handle string variable
        if isinstance(variable, str):
            variable = self.get_symbol(variable)
        
        # If no variable specified, use the first symbol found
        if variable is None:
            symbols = list(equation.free_symbols)
            if not symbols:
                raise ValueError("No variables found in the equation")
            variable = symbols[0]
        
        # Solve the equation
        solutions = sp.solve(equation, variable)
        return solutions
    
    def differentiate(self, 
                     expression: Union[sp.Expr, str], 
                     variable: Union[sp.Symbol, str], 
                     order: int = 1) -> sp.Expr:
        """
        Differentiate an expression with respect to a variable.
        
        Args:
            expression: SymPy expression or string representation
            variable: Variable to differentiate with respect to
            order: Order of differentiation
            
        Returns:
            Differentiated expression
        """
        # Handle string input
        if isinstance(expression, str):
            expression = sp.sympify(expression)
        
        # Handle string variable
        if isinstance(variable, str):
            variable = self.get_symbol(variable)
        
        # Differentiate the expression
        result = sp.diff(expression, variable, order)
        return result
    
    def integrate(self, 
                 expression: Union[sp.Expr, str], 
                 variable: Union[sp.Symbol, str], 
                 lower_bound: Optional[Union[sp.Expr, str, float]] = None, 
                 upper_bound: Optional[Union[sp.Expr, str, float]] = None) -> sp.Expr:
        """
        Integrate an expression with respect to a variable.
        
        Args:
            expression: SymPy expression or string representation
            variable: Variable to integrate with respect to
            lower_bound: Lower bound for definite integration
            upper_bound: Upper bound for definite integration
            
        Returns:
            Integrated expression
        """
        # Handle string input
        if isinstance(expression, str):
            expression = sp.sympify(expression)
        
        # Handle string variable
        if isinstance(variable, str):
            variable = self.get_symbol(variable)
        
        # Handle string bounds
        if isinstance(lower_bound, str):
            lower_bound = sp.sympify(lower_bound)
        if isinstance(upper_bound, str):
            upper_bound = sp.sympify(upper_bound)
        
        # Perform indefinite or definite integration
        if lower_bound is not None and upper_bound is not None:
            result = sp.integrate(expression, (variable, lower_bound, upper_bound))
        else:
            result = sp.integrate(expression, variable)
        
        return result
    
    def evaluate(self, 
                expression: Union[sp.Expr, str], 
                values: Dict[Union[str, sp.Symbol], Union[float, int, sp.Expr]]) -> Union[float, sp.Expr]:
        """
        Evaluate an expression with specific variable values.
        
        Args:
            expression: SymPy expression or string representation
            values: Dictionary of variable values
            
        Returns:
            Evaluated expression
        """
        # Handle string input
        if isinstance(expression, str):
            expression = sp.sympify(expression)
        
        # Convert string keys to symbols
        subs_dict = {}
        for key, value in values.items():
            if isinstance(key, str):
                subs_dict[self.get_symbol(key)] = value
            else:
                subs_dict[key] = value
        
        # Substitute values
        result = expression.subs(subs_dict)
        
        # Try to simplify
        try:
            result = sp.simplify(result)
        except:
            pass
        
        return result
    
    def solve_system(self, 
                    equations: List[Union[sp.Eq, str]], 
                    variables: Optional[List[Union[sp.Symbol, str]]] = None) -> Dict[sp.Symbol, sp.Expr]:
        """
        Solve a system of equations.
        
        Args:
            equations: List of SymPy equations or string representations
            variables: List of variables to solve for (if None, solver will try to determine)
            
        Returns:
            Dictionary mapping variables to their solutions
        """
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
        if variables is not None:
            processed_variables = []
            for var in variables:
                if isinstance(var, str):
                    processed_variables.append(self.get_symbol(var))
                else:
                    processed_variables.append(var)
        else:
            # Find all variables in the system
            all_symbols = set()
            for eq in processed_equations:
                all_symbols.update(eq.free_symbols)
            processed_variables = list(all_symbols)
        
        # Solve the system
        solution = sp.solve(processed_equations, processed_variables, dict=True)
        
        # Handle the case where solve returns a list of dictionaries
        if isinstance(solution, list):
            if not solution:
                return {}
            return solution[0]  # Return the first solution
        
        return solution
    
    def simplify(self, expression: Union[sp.Expr, str]) -> sp.Expr:
        """
        Simplify a mathematical expression.
        
        Args:
            expression: SymPy expression or string representation
            
        Returns:
            Simplified expression
        """
        # Handle string input
        if isinstance(expression, str):
            expression = sp.sympify(expression)
        
        # Apply various simplification techniques
        result = sp.simplify(expression)
        
        return result
    
    def factor(self, expression: Union[sp.Expr, str]) -> sp.Expr:
        """
        Factor a polynomial expression.
        
        Args:
            expression: SymPy expression or string representation
            
        Returns:
            Factored expression
        """
        # Handle string input
        if isinstance(expression, str):
            expression = sp.sympify(expression)
        
        # Factor the expression
        result = sp.factor(expression)
        
        return result
    
    def expand(self, expression: Union[sp.Expr, str]) -> sp.Expr:
        """
        Expand a mathematical expression.
        
        Args:
            expression: SymPy expression or string representation
            
        Returns:
            Expanded expression
        """
        # Handle string input
        if isinstance(expression, str):
            expression = sp.sympify(expression)
        
        # Expand the expression
        result = sp.expand(expression)
        
        return result
    
    def to_latex(self, expr: Any) -> str:
        """
        Convert a SymPy expression to LaTeX.
        
        Args:
            expr: SymPy expression, equation, or container of expressions
            
        Returns:
            LaTeX representation
        """
        return sp.latex(expr)
    
    def to_text(self, expr: Any) -> str:
        """
        Convert a SymPy expression to plain text.
        
        Args:
            expr: SymPy expression, equation, or container of expressions
            
        Returns:
            Plain text representation
        """
        text = str(expr)
        text = text.replace("**", "^")
        text = text.replace("*", "·")
        return text

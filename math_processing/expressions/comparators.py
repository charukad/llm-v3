"""
Expression Comparators - compare mathematical expressions for equivalence.

This module provides functionality for comparing mathematical expressions
to determine if they are equivalent, regardless of their specific form.
"""

import sympy as sp
from typing import Dict, Union, Any, Tuple, List
import random


class ExpressionComparator:
    """Comparator for mathematical expressions."""
    
    def __init__(self):
        """Initialize the expression comparator."""
        pass
    
    def is_equivalent(self, 
                     expr1: Union[sp.Expr, sp.Eq], 
                     expr2: Union[sp.Expr, sp.Eq],
                     method: str = "symbolic") -> Dict[str, Any]:
        """
        Check if two expressions are mathematically equivalent.
        
        Args:
            expr1: First SymPy expression or equation
            expr2: Second SymPy expression or equation
            method: Comparison method (symbolic, numerical, simplify)
            
        Returns:
            Dictionary containing:
                - equivalent: Boolean indicating if expressions are equivalent
                - method: Method used for comparison
                - explanation: Explanation of the comparison
                - error: Error message if an error occurred, None otherwise
        """
        try:
            # Check if both are equations or both are expressions
            if isinstance(expr1, sp.Eq) != isinstance(expr2, sp.Eq):
                return {
                    "equivalent": False,
                    "method": method,
                    "explanation": "One expression is an equation, the other is not",
                    "error": None
                }
            
            # Choose appropriate comparison method
            if method == "symbolic":
                equivalent, explanation = self._symbolic_comparison(expr1, expr2)
            elif method == "numerical":
                equivalent, explanation = self._numerical_comparison(expr1, expr2)
            elif method == "simplify":
                equivalent, explanation = self._simplify_comparison(expr1, expr2)
            else:
                # Default to symbolic if method not recognized
                equivalent, explanation = self._symbolic_comparison(expr1, expr2)
            
            return {
                "equivalent": equivalent,
                "method": method,
                "explanation": explanation,
                "error": None
            }
        except Exception as e:
            return {
                "equivalent": False,
                "method": method,
                "explanation": "",
                "error": str(e)
            }
    
    def _symbolic_comparison(self, 
                           expr1: Union[sp.Expr, sp.Eq], 
                           expr2: Union[sp.Expr, sp.Eq]) -> Tuple[bool, str]:
        """
        Compare expressions symbolically.
        
        Args:
            expr1: First SymPy expression or equation
            expr2: Second SymPy expression or equation
            
        Returns:
            Tuple containing:
                - Boolean indicating if expressions are equivalent
                - Explanation of the comparison
        """
        if isinstance(expr1, sp.Eq) and isinstance(expr2, sp.Eq):
            # For equations, rearrange to standard form: expr = 0
            expr1_std = sp.Eq(expr1.lhs - expr1.rhs, 0)
            expr2_std = sp.Eq(expr2.lhs - expr2.rhs, 0)
            
            # Check if the difference simplifies to zero
            diff = expr1_std.lhs - expr2_std.lhs
            diff_simplified = sp.simplify(diff)
            
            if diff_simplified == 0:
                return True, "The equations are symbolically equivalent"
            else:
                return False, "The equations are not symbolically equivalent"
        else:
            # For expressions, check if their difference simplifies to zero
            diff = expr1 - expr2
            diff_simplified = sp.simplify(diff)
            
            if diff_simplified == 0:
                return True, "The expressions are symbolically equivalent"
            else:
                return False, "The expressions are not symbolically equivalent"
    
    def _numerical_comparison(self, 
                            expr1: Union[sp.Expr, sp.Eq], 
                            expr2: Union[sp.Expr, sp.Eq]) -> Tuple[bool, str]:
        """
        Compare expressions numerically by evaluating at random points.
        
        Args:
            expr1: First SymPy expression or equation
            expr2: Second SymPy expression or equation
            
        Returns:
            Tuple containing:
                - Boolean indicating if expressions are equivalent
                - Explanation of the comparison
        """
        # Get all free symbols in both expressions
        if isinstance(expr1, sp.Eq) and isinstance(expr2, sp.Eq):
            expr1_symbols = expr1.lhs.free_symbols.union(expr1.rhs.free_symbols)
            expr2_symbols = expr2.lhs.free_symbols.union(expr2.rhs.free_symbols)
        else:
            expr1_symbols = expr1.free_symbols
            expr2_symbols = expr2.free_symbols
        
        all_symbols = expr1_symbols.union(expr2_symbols)
        
        # If there are no free symbols, use symbolic comparison
        if not all_symbols:
            return self._symbolic_comparison(expr1, expr2)
        
        # Test at multiple random points
        n_points = 5
        for _ in range(n_points):
            # Generate random values for all symbols
            test_point = {sym: random.uniform(-10, 10) for sym in all_symbols}
            
            try:
                # Evaluate both expressions at the test point
                if isinstance(expr1, sp.Eq) and isinstance(expr2, sp.Eq):
                    # For equations, check if both are satisfied or not satisfied
                    expr1_lhs_val = float(expr1.lhs.subs(test_point))
                    expr1_rhs_val = float(expr1.rhs.subs(test_point))
                    expr2_lhs_val = float(expr2.lhs.subs(test_point))
                    expr2_rhs_val = float(expr2.rhs.subs(test_point))
                    
                    expr1_satisfied = abs(expr1_lhs_val - expr1_rhs_val) < 1e-10
                    expr2_satisfied = abs(expr2_lhs_val - expr2_rhs_val) < 1e-10
                    
                    if expr1_satisfied != expr2_satisfied:
                        return False, f"The equations differ at point {test_point}"
                else:
                    # For expressions, check if their values are equal
                    expr1_val = float(expr1.subs(test_point))
                    expr2_val = float(expr2.subs(test_point))
                    
                    if abs(expr1_val - expr2_val) > 1e-10:
                        return False, f"The expressions differ at point {test_point}"
            except:
                # Skip points where evaluation fails (e.g., division by zero)
                continue
        
        return True, "The expressions are numerically equivalent at all tested points"
    
    def _simplify_comparison(self, 
                           expr1: Union[sp.Expr, sp.Eq], 
                           expr2: Union[sp.Expr, sp.Eq]) -> Tuple[bool, str]:
        """
        Compare expressions by simplifying each and checking for equality.
        
        Args:
            expr1: First SymPy expression or equation
            expr2: Second SymPy expression or equation
            
        Returns:
            Tuple containing:
                - Boolean indicating if expressions are equivalent
                - Explanation of the comparison
        """
        if isinstance(expr1, sp.Eq) and isinstance(expr2, sp.Eq):
            # For equations, compare the simplified equations
            expr1_lhs = sp.simplify(expr1.lhs - expr1.rhs)
            expr2_lhs = sp.simplify(expr2.lhs - expr2.rhs)
            
            if expr1_lhs == expr2_lhs:
                return True, "The simplified equations are identical"
            else:
                return False, "The simplified equations are different"
        else:
            # For expressions, compare the simplified expressions
            expr1_simp = sp.simplify(expr1)
            expr2_simp = sp.simplify(expr2)
            
            if expr1_simp == expr2_simp:
                return True, "The simplified expressions are identical"
            else:
                return False, "The simplified expressions are different"


def compare_expressions(
    expr1: Union[sp.Expr, sp.Eq],
    expr2: Union[sp.Expr, sp.Eq],
    method: str = "symbolic"
) -> Dict[str, Any]:
    """
    Compare two mathematical expressions for equivalence.
    
    This is the main function to use when importing this module.
    
    Args:
        expr1: First SymPy expression or equation
        expr2: Second SymPy expression or equation
        method: Comparison method (symbolic, numerical, simplify)
        
    Returns:
        Dictionary containing comparison results
    """
    comparator = ExpressionComparator()
    return comparator.is_equivalent(expr1, expr2, method)

"""
Expression Normalizer - standardizes mathematical expressions.

This module provides normalization functionality for mathematical expressions,
ensuring consistent representation regardless of how they were initially written.
"""

import sympy as sp
from typing import Dict, Union, Any, Optional


class ExpressionNormalizer:
    """Normalizer for mathematical expressions."""
    
    def __init__(self):
        """Initialize the expression normalizer."""
        pass
    
    def normalize(self, 
                 expr: Union[sp.Expr, sp.Eq], 
                 domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Normalize a SymPy expression or equation.
        
        Args:
            expr: SymPy expression or equation to normalize
            domain: Mathematical domain for specialized normalization
                    (algebra, calculus, etc.)
                    
        Returns:
            Dictionary containing:
                - success: Boolean indicating if normalization was successful
                - expression: Normalized expression if successful, None otherwise
                - error: Error message if unsuccessful, None otherwise
        """
        try:
            if domain:
                # Apply domain-specific normalization
                normalized_expr = self._domain_normalize(expr, domain)
            else:
                # Apply general normalization
                normalized_expr = self._general_normalize(expr)
                
            return {
                "success": True,
                "expression": normalized_expr,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "expression": None,
                "error": str(e)
            }
    
    def _general_normalize(self, expr: Union[sp.Expr, sp.Eq]) -> Union[sp.Expr, sp.Eq]:
        """
        Apply general normalization to a SymPy expression or equation.
        
        Args:
            expr: SymPy expression or equation
            
        Returns:
            Normalized expression or equation
        """
        if isinstance(expr, sp.Eq):
            # For equations, move all terms to the left side
            normalized = sp.Eq(expr.lhs - expr.rhs, 0)
            # Simplify the left side
            normalized = sp.Eq(sp.expand(normalized.lhs), 0)
        else:
            # For expressions, expand and simplify
            normalized = sp.expand(expr)
            
        return normalized
    
    def _domain_normalize(self, 
                         expr: Union[sp.Expr, sp.Eq], 
                         domain: str) -> Union[sp.Expr, sp.Eq]:
        """
        Apply domain-specific normalization.
        
        Args:
            expr: SymPy expression or equation
            domain: Mathematical domain
            
        Returns:
            Normalized expression or equation
        """
        # Start with general normalization
        normalized = self._general_normalize(expr)
        
        # Apply domain-specific normalization
        if domain.lower() == "algebra":
            normalized = self._normalize_algebra(normalized)
        elif domain.lower() == "calculus":
            normalized = self._normalize_calculus(normalized)
        elif domain.lower() == "linear_algebra":
            normalized = self._normalize_linear_algebra(normalized)
        
        return normalized
    
    def _normalize_algebra(self, expr: Union[sp.Expr, sp.Eq]) -> Union[sp.Expr, sp.Eq]:
        """
        Normalize algebraic expressions.
        
        Args:
            expr: SymPy expression or equation
            
        Returns:
            Normalized expression or equation
        """
        if isinstance(expr, sp.Eq):
            # For polynomial equations, collect terms
            lhs = sp.collect(expr.lhs, list(expr.lhs.free_symbols))
            normalized = sp.Eq(lhs, expr.rhs)
        else:
            # For polynomial expressions, collect terms
            normalized = sp.collect(expr, list(expr.free_symbols))
            
        return normalized
    
    def _normalize_calculus(self, expr: Union[sp.Expr, sp.Eq]) -> Union[sp.Expr, sp.Eq]:
        """
        Normalize calculus expressions.
        
        Args:
            expr: SymPy expression or equation
            
        Returns:
            Normalized expression or equation
        """
        # For calculus expressions, we might want different normalizations
        # This is a placeholder implementation
        return expr
    
    def _normalize_linear_algebra(self, expr: Union[sp.Expr, sp.Eq]) -> Union[sp.Expr, sp.Eq]:
        """
        Normalize linear algebra expressions.
        
        Args:
            expr: SymPy expression or equation
            
        Returns:
            Normalized expression or equation
        """
        # For linear algebra expressions, we might want different normalizations
        # This is a placeholder implementation
        return expr


def normalize_expression(
    expr: Union[sp.Expr, sp.Eq], 
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalize a mathematical expression.
    
    This is the main function to use when importing this module.
    
    Args:
        expr: SymPy expression or equation to normalize
        domain: Mathematical domain for specialized normalization
        
    Returns:
        Dictionary containing normalization results
    """
    normalizer = ExpressionNormalizer()
    return normalizer.normalize(expr, domain)

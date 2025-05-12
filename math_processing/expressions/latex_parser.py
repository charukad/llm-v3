"""
LaTeX Parser - converts LaTeX mathematical expressions to SymPy expressions.

This module handles the parsing of LaTeX mathematical notation into SymPy 
expressions that can be manipulated and computed with symbolically.
"""

import re
import sympy as sp
from sympy.parsing.latex import parse_latex
from typing import Dict, List, Optional, Tuple, Union, Any


class LaTeXParser:
    """Parser for LaTeX mathematical expressions."""
    
    def __init__(self):
        """Initialize the LaTeX parser."""
        # Common LaTeX symbols and their SymPy equivalents
        self.symbol_map = {
            r"\alpha": sp.Symbol('alpha'),
            r"\beta": sp.Symbol('beta'),
            r"\gamma": sp.Symbol('gamma'),
            r"\delta": sp.Symbol('delta'),
            r"\epsilon": sp.Symbol('epsilon'),
            r"\pi": sp.pi,
            r"\infty": sp.oo,
        }
        
        # Variables used throughout parsing
        self.variables = {}
        
    def parse(self, latex_str: str) -> Dict[str, Any]:
        """
        Parse a LaTeX string into a SymPy expression.
        
        Args:
            latex_str: LaTeX string representing a mathematical expression
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating if parsing was successful
                - expression: SymPy expression if successful, None otherwise
                - error: Error message if unsuccessful, None otherwise
                - variables: Dictionary of variables found in the expression
        """
        # Clean up the LaTeX string
        latex_str = self._preprocess_latex(latex_str)
        
        try:
            # Use SymPy's built-in LaTeX parser
            expr = parse_latex(latex_str)
            
            # Extract variables
            self._extract_variables(expr)
            
            return {
                "success": True,
                "expression": expr,
                "error": None,
                "variables": self.variables
            }
        except Exception as e:
            # If SymPy's parser fails, attempt custom parsing for common cases
            try:
                expr = self._custom_parse(latex_str)
                if expr is not None:
                    self._extract_variables(expr)
                    return {
                        "success": True,
                        "expression": expr,
                        "error": None,
                        "variables": self.variables
                    }
            except Exception as custom_error:
                # If custom parsing also fails, return the original error
                pass
            
            return {
                "success": False,
                "expression": None,
                "error": str(e),
                "variables": {}
            }
    
    def _preprocess_latex(self, latex_str: str) -> str:
        """
        Preprocess LaTeX string to handle common issues before parsing.
        
        Args:
            latex_str: Original LaTeX string
            
        Returns:
            Preprocessed LaTeX string
        """
        # Remove whitespace
        latex_str = latex_str.strip()
        
        # Handle implied multiplication (e.g., "2x" -> "2*x")
        # This is a common source of parsing errors
        pattern = r'(\d)([a-zA-Z])'
        latex_str = re.sub(pattern, r'\1 \2', latex_str)
        
        # Handle common LaTeX commands that might cause issues
        latex_str = latex_str.replace(r'\left(', '(').replace(r'\right)', ')')
        latex_str = latex_str.replace(r'\left[', '[').replace(r'\right]', ']')
        latex_str = latex_str.replace(r'\left{', '{').replace(r'\right}', '}')
        
        # Convert \frac{a}{b} to (a)/(b) for better parsing
        latex_str = self._convert_fractions(latex_str)
        
        return latex_str
    
    def _convert_fractions(self, latex_str: str) -> str:
        """
        Convert LaTeX fractions to a format better for parsing.
        
        Args:
            latex_str: LaTeX string with possible fractions
            
        Returns:
            Converted LaTeX string
        """
        # This is a simplified implementation
        # A full implementation would use a proper LaTeX parser
        
        # Find all instances of \frac{...}{...}
        frac_pattern = r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        
        # Function to process each match
        def replace_frac(match):
            numerator = match.group(1)
            denominator = match.group(2)
            return f"({numerator})/({denominator})"
        
        # Replace all fractions
        while r'\frac{' in latex_str:
            latex_str = re.sub(frac_pattern, replace_frac, latex_str)
        
        return latex_str
    
    def _custom_parse(self, latex_str: str) -> Optional[sp.Expr]:
        """
        Custom parser for common cases that SymPy's parser might fail on.
        
        Args:
            latex_str: Preprocessed LaTeX string
            
        Returns:
            SymPy expression if successful, None otherwise
        """
        # Example: Simple equation with =
        if '=' in latex_str:
            parts = latex_str.split('=')
            if len(parts) == 2:
                try:
                    lhs = parse_latex(parts[0])
                    rhs = parse_latex(parts[1])
                    return sp.Eq(lhs, rhs)
                except:
                    pass
        
        # If all custom parsing attempts fail, return None
        return None
    
    def _extract_variables(self, expr: sp.Expr) -> None:
        """
        Extract variables from a SymPy expression.
        
        Args:
            expr: SymPy expression
        """
        self.variables = {}
        for symbol in expr.free_symbols:
            self.variables[symbol.name] = symbol


def parse_math_expression(latex_string: str) -> Dict[str, Any]:
    """
    Parse a LaTeX string into a SymPy expression.
    
    This is the main function to use when importing this module.
    
    Args:
        latex_string: LaTeX string representing a mathematical expression
        
    Returns:
        Dictionary containing parsing results
    """
    parser = LaTeXParser()
    return parser.parse(latex_string)

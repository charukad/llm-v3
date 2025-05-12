"""
Mathematical expression parsing and conversion utilities.

This module provides functions to parse and convert mathematical expressions
between different formats (LaTeX, plaintext, and SymPy expressions).
"""

import logging
import re
from typing import Dict, Union, Optional, Any, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

logger = logging.getLogger(__name__)

# Common mathematical symbols and their SymPy equivalents
SYMBOL_MAP = {
    "\\pi": "pi",
    "\\infty": "oo",
    "\\sin": "sin",
    "\\cos": "cos",
    "\\tan": "tan",
    "\\log": "log",
    "\\ln": "ln",
    "\\exp": "exp",
    "\\sqrt": "sqrt",
    "\\frac": "frac",  # Special handling required
    "\\sum": "sum",    # Special handling required
    "\\int": "Integral",  # Special handling required
}

def clean_expression(expr: str) -> str:
    """
    Clean a mathematical expression string to prepare for parsing.
    
    Args:
        expr: Mathematical expression as a string
        
    Returns:
        Cleaned expression string
    """
    # Remove extra whitespace
    expr = expr.strip()
    expr = re.sub(r'\s+', ' ', expr)
    
    # Replace common LaTeX operations with plaintext equivalents
    expr = expr.replace("^", "**")  # Exponentiation
    expr = re.sub(r'(\d+)x', r'\1*x', expr)  # Implicit multiplication with numbers
    
    return expr

def parse_latex_to_sympy(latex_expr: str) -> sp.Expr:
    """
    Parse a LaTeX mathematical expression to a SymPy expression.
    
    Args:
        latex_expr: LaTeX mathematical expression
        
    Returns:
        SymPy expression
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        # For a full implementation, you would need a comprehensive LaTeX parser
        # This is a simplified version for demonstration
        
        # Replace LaTeX symbols with their SymPy equivalents
        expr = latex_expr
        for latex_symbol, sympy_symbol in SYMBOL_MAP.items():
            if latex_symbol in expr:
                if latex_symbol == "\\frac":
                    # Handle fractions - simplified approach
                    expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', expr)
                elif latex_symbol == "\\sqrt":
                    # Handle square roots
                    expr = re.sub(r'\\sqrt\{(.*?)\}', r'sqrt(\1)', expr)
                else:
                    expr = expr.replace(latex_symbol, sympy_symbol)
        
        # Clean expression
        expr = clean_expression(expr)
        
        # Parse with SymPy
        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed_expr = parse_expr(expr, transformations=transformations)
        
        return parsed_expr
        
    except Exception as e:
        logger.error(f"Failed to parse LaTeX expression '{latex_expr}': {e}")
        raise ValueError(f"Failed to parse LaTeX expression: {e}")

def parse_expression(expression_str: str, is_latex: bool = False) -> sp.Expr:
    """
    Parse a mathematical expression from string to SymPy expression.
    
    Args:
        expression_str: String representation of the mathematical expression
        is_latex: Whether the input is in LaTeX format
        
    Returns:
        SymPy expression
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        if is_latex:
            return parse_latex_to_sympy(expression_str)
        
        # Clean expression
        expr = clean_expression(expression_str)
        
        # Parse with SymPy
        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed_expr = parse_expr(expr, transformations=transformations)
        
        return parsed_expr
        
    except Exception as e:
        logger.error(f"Failed to parse expression '{expression_str}': {e}")
        raise ValueError(f"Failed to parse expression: {e}")

def sympy_to_latex(expr: sp.Expr) -> str:
    """
    Convert a SymPy expression to LaTeX format.
    
    Args:
        expr: SymPy expression
        
    Returns:
        LaTeX representation of the expression
    """
    return sp.latex(expr)

def parse_equation(equation_str: str, is_latex: bool = False) -> sp.Eq:
    """
    Parse an equation from string to SymPy equation.
    
    Args:
        equation_str: String representation of the equation
        is_latex: Whether the input is in LaTeX format
        
    Returns:
        SymPy equation
        
    Raises:
        ValueError: If parsing fails or if input is not an equation
    """
    try:
        if "=" not in equation_str:
            raise ValueError("Input is not an equation (no '=' found)")
        
        # Split by first equals sign
        left_str, right_str = equation_str.split("=", 1)
        
        # Parse both sides
        left_expr = parse_expression(left_str, is_latex=is_latex)
        right_expr = parse_expression(right_str, is_latex=is_latex)
        
        # Create equation
        return sp.Eq(left_expr, right_expr)
        
    except Exception as e:
        logger.error(f"Failed to parse equation '{equation_str}': {e}")
        raise ValueError(f"Failed to parse equation: {e}")

def extract_variables(expr: sp.Expr) -> set:
    """
    Extract all variables from a SymPy expression.
    
    Args:
        expr: SymPy expression
        
    Returns:
        Set of variables in the expression
    """
    return expr.free_symbols
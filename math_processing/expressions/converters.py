"""
Format Converters - convert between different mathematical expression formats.

This module handles conversion between different representation formats for 
mathematical expressions, including LaTeX, SymPy, and other formats.
"""

import sympy as sp
import re
from typing import Dict, Union, Any, Optional


class FormatConverter:
    """Converter between different mathematical expression formats."""
    
    def __init__(self):
        """Initialize the format converter."""
        pass
    
    def sympy_to_latex(self, expr: Union[sp.Expr, sp.Eq]) -> str:
        """
        Convert a SymPy expression to LaTeX.
        
        Args:
            expr: SymPy expression or equation
            
        Returns:
            LaTeX string representation
        """
        return sp.latex(expr)
    
    def sympy_to_text(self, expr: Union[sp.Expr, sp.Eq]) -> str:
        """
        Convert a SymPy expression to plain text.
        
        Args:
            expr: SymPy expression or equation
            
        Returns:
            Plain text representation
        """
        # Use SymPy's string representation
        text = str(expr)
        
        # Clean up some common patterns to make it more readable
        text = text.replace("**", "^")
        text = text.replace("*", "·")
        
        return text
    
    def sympy_to_python(self, expr: Union[sp.Expr, sp.Eq]) -> str:
        """
        Convert a SymPy expression to Python code.
        
        Args:
            expr: SymPy expression or equation
            
        Returns:
            Python code representation
        """
        # We can use SymPy's built-in Python code printer
        from sympy.printing.python import PythonPrinter
        
        printer = PythonPrinter()
        return printer.doprint(expr)
    
    def format_step_by_step(self, 
                           steps: list, 
                           format_type: str = "latex") -> Dict[str, Any]:
        """
        Format a list of solution steps in the specified format.
        
        Args:
            steps: List of solution steps, where each step is a SymPy expression
            format_type: Output format (latex, text, html)
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating if formatting was successful
                - steps: List of formatted steps
                - error: Error message if unsuccessful, None otherwise
        """
        try:
            formatted_steps = []
            
            for step in steps:
                if format_type.lower() == "latex":
                    formatted = self.sympy_to_latex(step.get("expression"))
                elif format_type.lower() == "text":
                    formatted = self.sympy_to_text(step.get("expression"))
                else:
                    # Default to LaTeX if format not recognized
                    formatted = self.sympy_to_latex(step.get("expression"))
                
                formatted_step = {
                    "expression": formatted,
                    "explanation": step.get("explanation", "")
                }
                
                formatted_steps.append(formatted_step)
            
            return {
                "success": True,
                "steps": formatted_steps,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "steps": [],
                "error": str(e)
            }


def convert_expression(
    expr: Union[sp.Expr, sp.Eq],
    to_format: str
) -> Dict[str, Any]:
    """
    Convert an expression to the specified format.
    
    This is a main function to use when importing this module.
    
    Args:
        expr: SymPy expression or equation to convert
        to_format: Target format (latex, text, python)
        
    Returns:
        Dictionary containing conversion results
    """
    converter = FormatConverter()
    
    try:
        if to_format.lower() == "latex":
            result = converter.sympy_to_latex(expr)
        elif to_format.lower() == "text":
            result = converter.sympy_to_text(expr)
        elif to_format.lower() == "python":
            result = converter.sympy_to_python(expr)
        else:
            return {
                "success": False,
                "result": None,
                "error": f"Unsupported format: {to_format}"
            }
        
        return {
            "success": True,
            "result": result,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e)
        }

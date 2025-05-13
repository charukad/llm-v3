"""
LaTeX generator for mathematical expressions.

This module converts structural representations of mathematical expressions
to LaTeX format.
"""
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def generate_latex(structure: Dict[str, Any]) -> str:
    """
    Generate LaTeX from a structural representation of a mathematical expression.
    
    Args:
        structure: Dictionary containing expression structure
        
    Returns:
        LaTeX representation of the expression
    """
    try:
        structure_type = structure.get("type", "unknown")
        
        if structure_type == "expression":
            return generate_expression_latex(structure)
        elif structure_type == "multi_line":
            return generate_multi_line_latex(structure)
        elif structure_type == "empty":
            return ""
        elif structure_type == "error":
            logger.error(f"Error in structure: {structure.get('error', 'Unknown error')}")
            return "\\text{Error in expression}"
        else:
            logger.warning(f"Unknown structure type: {structure_type}")
            return f"\\text{{Unknown: {structure_type}}}"
        
    except Exception as e:
        logger.error(f"Error generating LaTeX: {str(e)}")
        return f"\\text{{Error: {str(e)}}}"

def generate_expression_latex(structure: Dict[str, Any]) -> str:
    """
    Generate LaTeX for an expression structure.
    
    Args:
        structure: Dictionary containing expression structure
        
    Returns:
        LaTeX representation of the expression
    """
    elements = structure.get("elements", [])
    latex_parts = []
    
    for element in elements:
        element_type = element.get("type", "unknown")
        
        if element_type == "symbol":
            # Handle special symbols
            value = element.get("value", "")
            latex_parts.append(get_symbol_latex(value))
            
        elif element_type == "superscript":
            base = element.get("base", {})
            exponent = element.get("exponent", {})
            
            base_latex = generate_element_latex(base)
            exponent_latex = generate_element_latex(exponent)
            
            latex_parts.append(f"{base_latex}^{{{exponent_latex}}}")
            
        elif element_type == "subscript":
            base = element.get("base", {})
            subscript = element.get("subscript", {})
            
            base_latex = generate_element_latex(base)
            subscript_latex = generate_element_latex(subscript)
            
            latex_parts.append(f"{base_latex}_{{{subscript_latex}}}")
            
        elif element_type == "fraction":
            numerator = element.get("numerator", {})
            denominator = element.get("denominator", {})
            
            numerator_latex = generate_element_latex(numerator)
            denominator_latex = generate_element_latex(denominator)
            
            latex_parts.append(f"\\frac{{{numerator_latex}}}{{{denominator_latex}}}")
            
        else:
            logger.warning(f"Unknown element type: {element_type}")
            latex_parts.append(f"\\text{{Unknown}}")
    
    return "".join(latex_parts)

def generate_multi_line_latex(structure: Dict[str, Any]) -> str:
    """
    Generate LaTeX for a multi-line structure.
    
    Args:
        structure: Dictionary containing multi-line structure
        
    Returns:
        LaTeX representation of the multi-line expression
    """
    lines = structure.get("lines", [])
    latex_parts = []
    
    for line in lines:
        line_latex = generate_latex(line)
        latex_parts.append(line_latex)
    
    return " \\\\ ".join(latex_parts)

def generate_element_latex(element: Dict[str, Any]) -> str:
    """
    Generate LaTeX for a single element.
    
    Args:
        element: Dictionary containing element data
        
    Returns:
        LaTeX representation of the element
    """
    element_type = element.get("type", "unknown")
    
    if element_type == "symbol":
        return get_symbol_latex(element.get("value", ""))
    elif element_type == "expression":
        # For simple expressions that are just values
        if "value" in element:
            return element["value"]
        # For structured expressions
        if "elements" in element:
            return generate_expression_latex(element)
    else:
        logger.warning(f"Unknown element type in generate_element_latex: {element_type}")
        return f"\\text{{Unknown}}"

def get_symbol_latex(symbol: str) -> str:
    """
    Convert a symbol to its LaTeX representation.
    
    Args:
        symbol: Symbol text
        
    Returns:
        LaTeX representation of the symbol
    """
    # Map of special symbols to LaTeX commands
    symbol_map = {
        "+": "+",
        "-": "-",
        "*": "\\cdot ",
        "/": "/",
        "=": "=",
        ">": ">",
        "<": "<",
        "≥": "\\geq ",
        "≤": "\\leq ",
        "∞": "\\infty ",
        "π": "\\pi ",
        "θ": "\\theta ",
        "α": "\\alpha ",
        "β": "\\beta ",
        "γ": "\\gamma ",
        "δ": "\\delta ",
        "ε": "\\epsilon ",
        "λ": "\\lambda ",
        "σ": "\\sigma ",
        "∫": "\\int ",
        "∑": "\\sum ",
        "∏": "\\prod ",
        "√": "\\sqrt",
        "∂": "\\partial ",
        "Δ": "\\Delta ",
        "→": "\\to "
    }
    
    return symbol_map.get(symbol, symbol)

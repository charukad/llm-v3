"""
Tests for the LaTeX parser.
"""

import pytest
import sympy as sp
from math_processing.expressions.latex_parser import parse_math_expression, LaTeXParser


class TestLaTeXParser:
    """Test the LaTeX parser functionality."""
    
    def test_simple_expression(self):
        """Test parsing a simple algebraic expression."""
        result = parse_math_expression("x^2 + 2x + 1")
        
        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["expression"], sp.Expr)
        
        # Verify the parsed expression is correct
        x = sp.Symbol('x')
        expected = x**2 + 2*x + 1
        difference = sp.simplify(result["expression"] - expected)
        assert difference == 0
    
    def test_equation_parsing(self):
        """Test parsing an equation."""
        result = parse_math_expression("x^2 + 2x + 1 = 0")
        
        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["expression"], sp.Eq)
        
        # Verify the parsed equation is correct
        x = sp.Symbol('x')
        expected = sp.Eq(x**2 + 2*x + 1, 0)
        
        assert result["expression"].lhs == expected.lhs
        assert result["expression"].rhs == expected.rhs
    
    def test_fraction_parsing(self):
        """Test parsing a LaTeX fraction."""
        result = parse_math_expression("\\frac{x^2 + 1}{x - 1}")
        
        assert result["success"] is True
        assert result["error"] is None
        
        # Verify the parsed expression is correct
        x = sp.Symbol('x')
        expected = (x**2 + 1) / (x - 1)
        difference = sp.simplify(result["expression"] - expected)
        assert difference == 0
    
    def test_trigonometric_functions(self):
        """Test parsing trigonometric functions."""
        result = parse_math_expression("\\sin(x) + \\cos(x)")
        
        assert result["success"] is True
        assert result["error"] is None
        
        # Verify the parsed expression is correct
        x = sp.Symbol('x')
        expected = sp.sin(x) + sp.cos(x)
        difference = sp.simplify(result["expression"] - expected)
        assert difference == 0
    
    def test_invalid_expression(self):
        """Test parsing an invalid expression."""
        result = parse_math_expression("x +/ 2")
        
        assert result["success"] is False
        assert result["error"] is not None
        assert result["expression"] is None

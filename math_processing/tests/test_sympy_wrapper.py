"""
Tests for the SymPy wrapper.
"""

import pytest
import sympy as sp
from math_processing.computation.sympy_wrapper import SymbolicProcessor


class TestSymbolicProcessor:
    """Test the symbolic processor functionality."""
    
    def setup_method(self):
        """Set up the test environment."""
        self.processor = SymbolicProcessor()
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
    
    def test_solve_equation(self):
        """Test solving an equation."""
        # Test with a quadratic equation
        equation = sp.Eq(self.x**2 + 2*self.x + 1, 0)
        solutions = self.processor.solve_equation(equation, self.x)
        
        # Expected solution: x = -1
        assert len(solutions) == 1
        assert sp.simplify(solutions[0] + 1) == 0
        
        # Test with a string equation
        solutions = self.processor.solve_equation("x^2 + 2*x + 1 = 0", "x")
        assert len(solutions) == 1
        assert sp.simplify(solutions[0] + 1) == 0
    
    def test_differentiate(self):
        """Test differentiation."""
        # Test with a polynomial
        expr = self.x**3 + 2*self.x**2 - 5*self.x + 3
        derivative = self.processor.differentiate(expr, self.x)
        
        # Expected derivative: 3x² + 4x - 5
        expected = 3*self.x**2 + 4*self.x - 5
        assert sp.simplify(derivative - expected) == 0
        
        # Test with a string expression
        derivative = self.processor.differentiate("x^3 + 2*x^2 - 5*x + 3", "x")
        assert sp.simplify(derivative - expected) == 0
        
        # Test with higher order
        second_derivative = self.processor.differentiate(expr, self.x, 2)
        expected_second = 6*self.x + 4
        assert sp.simplify(second_derivative - expected_second) == 0
    
    def test_integrate(self):
        """Test integration."""
        # Test with a polynomial
        expr = 3*self.x**2 + 2*self.x
        integral = self.processor.integrate(expr, self.x)
        
        # Expected integral: x³ + x² + C
        expected = self.x**3 + self.x**2
        assert sp.simplify(integral - expected) == 0
        
        # Test with a string expression
        integral = self.processor.integrate("3*x^2 + 2*x", "x")
        assert sp.simplify(integral - expected) == 0
        
        # Test definite integral
        definite_integral = self.processor.integrate(expr, self.x, 0, 1)
        # Expected: (1³ + 1²) - (0³ + 0²) = 2
        assert definite_integral == 2
    
    def test_evaluate(self):
        """Test expression evaluation."""
        # Test with a simple expression
        expr = self.x**2 + 2*self.x*self.y + self.y**2
        result = self.processor.evaluate(expr, {'x': 3, 'y': 4})
        
        # Expected: 3² + 2*3*4 + 4² = 9 + 24 + 16 = 49
        assert result == 49
        
        # Test with a string expression
        result = self.processor.evaluate("x^2 + 2*x*y + y^2", {'x': 3, 'y': 4})
        assert result == 49
    
    def test_simplify(self):
        """Test expression simplification."""
        # Test with a rational expression
        expr = (self.x**2 - 1) / (self.x - 1)
        simplified = self.processor.simplify(expr)
        
        # Expected: x + 1 (after cancellation)
        expected = self.x + 1
        assert sp.simplify(simplified - expected) == 0
        
        # Test with a string expression
        simplified = self.processor.simplify("(x^2 - 1) / (x - 1)")
        assert sp.simplify(simplified - expected) == 0
    
    def test_factor(self):
        """Test polynomial factorization."""
        # Test with a polynomial
        expr = self.x**2 - 1
        factored = self.processor.factor(expr)
        
        # Expected: (x - 1)(x + 1)
        expected = (self.x - 1) * (self.x + 1)
        assert sp.simplify(factored - expected) == 0
        
        # Test with a string expression
        factored = self.processor.factor("x^2 - 1")
        assert sp.simplify(factored - expected) == 0
    
    def test_expand(self):
        """Test expression expansion."""
        # Test with a factored expression
        expr = (self.x + 1) * (self.x - 1)
        expanded = self.processor.expand(expr)
        
        # Expected: x² - 1
        expected = self.x**2 - 1
        assert sp.simplify(expanded - expected) == 0
        
        # Test with a string expression
        expanded = self.processor.expand("(x + 1) * (x - 1)")
        assert sp.simplify(expanded - expected) == 0

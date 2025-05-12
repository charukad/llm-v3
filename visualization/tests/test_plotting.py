"""
Tests for the plotting modules in the Visualization component.
"""

import unittest
import os
import sys
import numpy as np
import sympy as sp
import tempfile

# Add parent directory to Python path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from visualization.plotting.plot_2d import plot_function_2d, plot_multiple_functions_2d
from visualization.plotting.plot_3d import plot_function_3d, plot_parametric_3d
from visualization.plotting.statistical import plot_histogram, plot_scatter

class TestPlot2D(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def test_plot_function_2d_with_string(self):
        """Test plotting a 2D function with a string expression."""
        # Test with a string expression
        result = plot_function_2d(
            function_expr="sin(x)",
            x_range=(-5, 5),
            save_path=os.path.join(self.test_dir, "test_plot_2d_string.png")
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["file_path"]))
    
    def test_plot_function_2d_with_sympy(self):
        """Test plotting a 2D function with a SymPy expression."""
        # Test with a SymPy expression
        x = sp.Symbol('x')
        expr = sp.sin(x) + sp.cos(x)
        
        result = plot_function_2d(
            function_expr=expr,
            x_range=(-5, 5),
            save_path=os.path.join(self.test_dir, "test_plot_2d_sympy.png")
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["file_path"]))
    
    def test_plot_multiple_functions_2d(self):
        """Test plotting multiple 2D functions."""
        x = sp.Symbol('x')
        functions = [sp.sin(x), sp.cos(x), x**2 / 10]
        
        result = plot_multiple_functions_2d(
            functions=functions,
            labels=["sin(x)", "cos(x)", "x²/10"],
            x_range=(-5, 5),
            save_path=os.path.join(self.test_dir, "test_plot_multiple_2d.png")
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["file_path"]))
    
    def test_plot_function_2d_base64(self):
        """Test plotting a 2D function with base64 output."""
        # Test with base64 output (no save path)
        result = plot_function_2d(
            function_expr="sin(x)",
            x_range=(-5, 5)
        )
        
        self.assertTrue(result["success"])
        self.assertIn("base64_image", result)
        self.assertTrue(len(result["base64_image"]) > 0)

class TestPlot3D(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def test_plot_function_3d(self):
        """Test plotting a 3D function."""
        # Test with a string expression
        result = plot_function_3d(
            function_expr="sin(sqrt(x**2 + y**2))",
            x_range=(-5, 5),
            y_range=(-5, 5),
            save_path=os.path.join(self.test_dir, "test_plot_3d.png")
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["file_path"]))
    
    def test_plot_parametric_3d(self):
        """Test plotting a 3D parametric curve."""
        # Test with string expressions
        result = plot_parametric_3d(
            x_expr="cos(t)",
            y_expr="sin(t)",
            z_expr="t",
            t_range=(0, 6*np.pi),
            save_path=os.path.join(self.test_dir, "test_plot_parametric_3d.png")
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["file_path"]))

class TestStatisticalPlotting(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def test_plot_histogram(self):
        """Test plotting a histogram."""
        # Generate random data
        data = np.random.normal(0, 1, 1000)
        
        result = plot_histogram(
            data=data,
            bins=30,
            title="Test Histogram",
            save_path=os.path.join(self.test_dir, "test_histogram.png")
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["file_path"]))
    
    def test_plot_scatter(self):
        """Test plotting a scatter plot."""
        # Generate random data
        x_data = np.random.uniform(-5, 5, 100)
        y_data = x_data * 2 + np.random.normal(0, 1, 100)
        
        result = plot_scatter(
            x_data=x_data,
            y_data=y_data,
            title="Test Scatter Plot",
            show_regression=True,
            save_path=os.path.join(self.test_dir, "test_scatter.png")
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["file_path"]))

if __name__ == "__main__":
    unittest.main()

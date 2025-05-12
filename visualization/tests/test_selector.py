"""
Tests for the Visualization Selector.
"""

import unittest
import os
import sys

# Add parent directory to Python path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from visualization.selection.context_analyzer import VisualizationSelector

class TestVisualizationSelector(unittest.TestCase):
    def setUp(self):
        # Initialize visualization selector
        self.selector = VisualizationSelector()
    
    def test_select_2d_function(self):
        """Test selecting a 2D function visualization."""
        # Create test context
        context = {
            "expression": "sin(x)",
            "domain": "calculus"
        }
        
        # Select visualization
        result = self.selector.select_visualization(context)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["recommended_visualization"]["type"], "function_2d")
    
    def test_select_3d_function(self):
        """Test selecting a 3D function visualization."""
        # Create test context
        context = {
            "expression": "sin(x*y)",
            "domain": "calculus"
        }
        
        # Select visualization
        result = self.selector.select_visualization(context)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["recommended_visualization"]["type"], "function_3d")
    
    def test_select_derivative(self):
        """Test selecting a derivative visualization."""
        # Create test context
        context = {
            "expression": "x^3",
            "operation": "differentiate",
            "domain": "calculus"
        }
        
        # Select visualization
        result = self.selector.select_visualization(context)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["recommended_visualization"]["type"], "derivative")
    
    def test_select_histogram(self):
        """Test selecting a histogram visualization."""
        # Create test context
        context = {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "data_type": "histogram"
        }
        
        # Select visualization
        result = self.selector.select_visualization(context)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["recommended_visualization"]["type"], "histogram")
    
    def test_no_suitable_visualization(self):
        """Test handling a context with no suitable visualization."""
        # Create test context with no relevant information
        context = {
            "unrelated": "data"
        }
        
        # Select visualization
        result = self.selector.select_visualization(context)
        
        # Check result
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("No suitable visualization found", result["error"])
        self.assertIn("fallback", result)

if __name__ == "__main__":
    unittest.main()

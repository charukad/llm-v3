"""
Test script for all visualization types in SuperVisualizationAgent.

This script tests each visualization type to verify that they work correctly.
"""

import os
import sys
import unittest
from typing import Dict, Any
import json

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.agent.super_viz_agent import SuperVisualizationAgent

class TestSuperVisualizationAgent(unittest.TestCase):
    """Test suite for SuperVisualizationAgent."""
    
    def setUp(self):
        """Set up the test environment."""
        self.test_dir = "test_visualizations"
        os.makedirs(self.test_dir, exist_ok=True)
        self.agent = SuperVisualizationAgent({
            "storage_dir": self.test_dir,
            "use_database": False
        })
    
    def tearDown(self):
        """Clean up test files after tests."""
        # Comment this out if you want to keep the visualization files
        # import shutil
        # shutil.rmtree(self.test_dir, ignore_errors=True)
        pass
    
    def create_message(self, vis_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a message for the visualization agent."""
        return {
            "header": {
                "message_id": "test",
                "sender": "test",
                "recipient": "visualization_agent",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": vis_type,
                "parameters": params
            }
        }
    
    def test_function_2d(self):
        """Test 2D function plotting."""
        message = self.create_message("function_2d", {
            "expression": "sin(x)",
            "x_range": [-3.14, 3.14],
            "title": "Sine Function",
            "filename": "test_function_2d.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_function_2d.png")))
        print(f"Function 2D test result: {result}")
    
    def test_function_3d(self):
        """Test 3D function plotting."""
        message = self.create_message("function_3d", {
            "expression": "sin(x)*cos(y)",
            "x_range": [-3.14, 3.14],
            "y_range": [-3.14, 3.14],
            "title": "3D Surface",
            "filename": "test_function_3d.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_function_3d.png")))
        print(f"Function 3D test result: {result}")
    
    def test_histogram(self):
        """Test histogram plotting."""
        import numpy as np
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000).tolist()
        
        message = self.create_message("histogram", {
            "data": data,
            "bins": 30,
            "title": "Normal Distribution",
            "filename": "test_histogram.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_histogram.png")))
        print(f"Histogram test result: {result}")
    
    def test_scatter(self):
        """Test scatter plot."""
        import numpy as np
        np.random.seed(42)
        x_data = np.random.uniform(-10, 10, 50).tolist()
        y_data = [x + np.random.normal(0, 2) for x in x_data]
        
        message = self.create_message("scatter", {
            "x_data": x_data,
            "y_data": y_data,
            "title": "Scatter Plot",
            "show_regression": True,
            "filename": "test_scatter.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_scatter.png")))
        print(f"Scatter test result: {result}")
    
    def test_boxplot(self):
        """Test boxplot."""
        import numpy as np
        np.random.seed(42)
        data = [
            np.random.normal(0, 1, 100).tolist(),
            np.random.normal(2, 1.5, 100).tolist(),
            np.random.normal(-1, 0.5, 100).tolist()
        ]
        
        message = self.create_message("boxplot", {
            "data": data,
            "title": "Box Plot Comparison",
            "filename": "test_boxplot.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_boxplot.png")))
        print(f"Boxplot test result: {result}")
    
    def test_violin(self):
        """Test violin plot."""
        import numpy as np
        np.random.seed(42)
        data = [
            np.random.normal(0, 1, 100).tolist(),
            np.random.normal(2, 1.5, 100).tolist(),
            np.random.normal(-1, 0.5, 100).tolist()
        ]
        
        message = self.create_message("violin", {
            "data": data,
            "title": "Violin Plot Comparison",
            "filename": "test_violin.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_violin.png")))
        print(f"Violin test result: {result}")
    
    def test_bar(self):
        """Test bar chart."""
        message = self.create_message("bar", {
            "values": [25, 40, 30, 55, 15],
            "labels": ["Category A", "Category B", "Category C", "Category D", "Category E"],
            "title": "Bar Chart",
            "filename": "test_bar.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_bar.png")))
        print(f"Bar chart test result: {result}")
    
    def test_pie(self):
        """Test pie chart."""
        message = self.create_message("pie", {
            "values": [25, 40, 30, 5],
            "labels": ["Category A", "Category B", "Category C", "Other"],
            "title": "Pie Chart",
            "filename": "test_pie.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_pie.png")))
        print(f"Pie chart test result: {result}")
    
    def test_heatmap(self):
        """Test heatmap."""
        import numpy as np
        np.random.seed(42)
        data = np.random.rand(10, 10).tolist()
        
        message = self.create_message("heatmap", {
            "data": data,
            "title": "Heatmap",
            "filename": "test_heatmap.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_heatmap.png")))
        print(f"Heatmap test result: {result}")
    
    def test_contour(self):
        """Test contour plot."""
        message = self.create_message("contour", {
            "expression": "x**2 + y**2",
            "x_range": [-5, 5],
            "y_range": [-5, 5],
            "title": "Contour Plot",
            "levels": 20,
            "filled": True,
            "filename": "test_contour.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_contour.png")))
        print(f"Contour plot test result: {result}")
    
    def test_complex_function(self):
        """Test complex function plot."""
        message = self.create_message("complex_function", {
            "expression": "z**2",
            "title": "Complex Function Visualization",
            "filename": "test_complex.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_complex.png")))
        print(f"Complex function test result: {result}")
    
    def test_slope_field(self):
        """Test slope field plot."""
        message = self.create_message("slope_field", {
            "expression": "y",  # dy/dx = y (exponential growth)
            "x_range": [-5, 5],
            "y_range": [-5, 5],
            "title": "Slope Field",
            "solutions": [-2, -1, 0, 1, 2],
            "filename": "test_slope_field.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_slope_field.png")))
        print(f"Slope field test result: {result}")
    
    def test_time_series(self):
        """Test time series plot."""
        import numpy as np
        np.random.seed(42)
        data = np.cumsum(np.random.normal(0, 1, 100)).tolist()
        times = list(range(len(data)))
        
        message = self.create_message("time_series", {
            "data": data,
            "times": times,
            "title": "Time Series Plot",
            "filename": "test_time_series.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_time_series.png")))
        print(f"Time series test result: {result}")
    
    def test_correlation_matrix(self):
        """Test correlation matrix plot."""
        data = [
            [1.0, 0.7, 0.2, -0.1, 0.5],
            [0.7, 1.0, 0.3, 0.0, 0.6],
            [0.2, 0.3, 1.0, 0.8, 0.3],
            [-0.1, 0.0, 0.8, 1.0, 0.2],
            [0.5, 0.6, 0.3, 0.2, 1.0]
        ]
        
        message = self.create_message("correlation_matrix", {
            "data": data,
            "labels": ["Var 1", "Var 2", "Var 3", "Var 4", "Var 5"],
            "title": "Correlation Matrix",
            "filename": "test_correlation_matrix.png"
        })
        
        result = self.agent.process_message(message)
        self.assertTrue(result.get("success", False))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_correlation_matrix.png")))
        print(f"Correlation matrix test result: {result}")
    
    def test_get_capabilities(self):
        """Test the get_capabilities method."""
        capabilities = self.agent.get_capabilities()
        self.assertIsInstance(capabilities, dict)
        self.assertIn("supported_types", capabilities)
        print(f"Agent capabilities: {json.dumps(capabilities, indent=2)}")
        
        # Check that all our test methods are covered in the supported_types
        test_methods = [method for method in dir(self) if method.startswith("test_") and method not in ["test_get_capabilities"]]
        for method in test_methods:
            vis_type = method[5:]  # Remove "test_" prefix
            if vis_type in ["function_2d", "function_3d"]:  # Special cases
                self.assertIn(vis_type, capabilities["supported_types"])
            else:
                # The method name should have a corresponding type in supported_types
                self.assertTrue(any(t == vis_type for t in capabilities["supported_types"]),
                               f"Visualization type {vis_type} not found in capabilities")

if __name__ == '__main__':
    unittest.main() 
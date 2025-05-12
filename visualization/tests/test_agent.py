"""
Tests for the Visualization Agent.
"""

import unittest
import os
import sys
import json
import tempfile
import uuid

# Add parent directory to Python path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from visualization.agent.viz_agent import VisualizationAgent
from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent

class TestVisualizationAgent(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize agent with test configuration
        self.agent = VisualizationAgent({
            "storage_dir": self.test_dir,
            "use_database": False
        })
    
    def test_process_message_2d_function(self):
        """Test processing a 2D function visualization request."""
        # Create test message
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "test",
                "recipient": "visualization_agent",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": "function_2d",
                "parameters": {
                    "expression": "sin(x)",
                    "x_range": (-5, 5)
                }
            }
        }
        
        # Process message
        result = self.agent.process_message(message)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["plot_type"], "2d_function")
        self.assertTrue(os.path.exists(result["file_path"]))
    
    def test_process_message_invalid_type(self):
        """Test processing a request with an invalid visualization type."""
        # Create test message
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "test",
                "recipient": "visualization_agent",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": "invalid_type",
                "parameters": {}
            }
        }
        
        # Process message
        result = self.agent.process_message(message)
        
        # Check result
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Unsupported visualization type", result["error"])
    
    def test_get_capabilities(self):
        """Test retrieving agent capabilities."""
        capabilities = self.agent.get_capabilities()
        
        # Check capabilities
        self.assertEqual(capabilities["agent_type"], "visualization")
        self.assertIsInstance(capabilities["supported_types"], list)
        self.assertIn("function_2d", capabilities["supported_types"])

class TestAdvancedVisualizationAgent(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize agent with test configuration
        self.agent = AdvancedVisualizationAgent({
            "storage_dir": self.test_dir,
            "use_database": False
        })
    
    def test_process_message_derivative(self):
        """Test processing a derivative visualization request."""
        # Create test message
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "test",
                "recipient": "advanced_visualization_agent",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": "derivative",
                "parameters": {
                    "expression": "x^2",
                    "variable": "x",
                    "x_range": (-5, 5)
                }
            }
        }
        
        # Process message
        result = self.agent.process_message(message)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertIn("file_path", result)
        self.assertTrue(os.path.exists(result["file_path"]))
    
    def test_get_capabilities(self):
        """Test retrieving agent capabilities."""
        capabilities = self.agent.get_capabilities()
        
        # Check capabilities
        self.assertEqual(capabilities["agent_type"], "advanced_visualization")
        self.assertIsInstance(capabilities["supported_types"], list)
        self.assertIn("derivative", capabilities["supported_types"])
        self.assertIsInstance(capabilities["advanced_features"], list)

if __name__ == "__main__":
    unittest.main()

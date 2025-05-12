"""
End-to-end integration test for the Mathematical Multimodal LLM System.

This module tests the complete end-to-end workflow from input processing
to response generation.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
import requests
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from orchestration.workflow.end_to_end_workflows import EndToEndWorkflowManager
from orchestration.agents.registry import AgentRegistry
from multimodal.context.context_manager import ContextManager

class TestEndToEndWorkflow(unittest.TestCase):
    """Integration test for end-to-end workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Initialize registry and context manager
        cls.registry = AgentRegistry()
        cls.context_manager = ContextManager()
        
        # Create mocks for required components
        cls._setup_mocks()
        
        # Initialize workflow manager with mocked components
        cls.workflow_manager = EndToEndWorkflowManager(cls.registry)
        
        # Create test files
        cls.test_files = cls._create_test_files()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test files
        for file_path in cls.test_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @classmethod
    def _setup_mocks(cls):
        """Set up mock components for testing."""
        # Mock Core LLM Agent
        core_llm_mock = MagicMock()
        core_llm_mock.generate_response.return_value = {
            "success": True,
            "response": "This is a mock response from the Core LLM Agent.",
            "processing_time_ms": 100
        }
        
        # Mock Math Computation Agent
        math_agent_mock = MagicMock()
        math_agent_mock.process_expression.return_value = {
            "success": True,
            "result": "4x",
            "latex_result": "4x",
            "steps": [
                {"description": "Apply the power rule", "latex": "\\frac{d}{dx}[x^2] = 2x"},
                {"description": "Apply the product rule", "latex": "\\frac{d}{dx}[x^2 \\cdot 2] = 2x \\cdot 2 = 4x"}
            ],
            "processing_time_ms": 50
        }
        
        # Mock Input Processor
        input_processor_mock = MagicMock()
        input_processor_mock.process_input.return_value = {
            "success": True,
            "input_type": "text",
            "text": "Find the derivative of 2x^2",
            "processing_time_ms": 20
        }
        
        # Mock Content Router
        content_router_mock = MagicMock()
        content_router_mock.route_content.return_value = {
            "success": True,
            "source_type": "text",
            "agent_type": "core_llm",
            "processing_time_ms": 10
        }
        
        # Register mocks in registry
        cls.registry.register_agent(
            agent_id="core_llm_agent",
            agent_info={
                "name": "Core LLM Agent",
                "instance": core_llm_mock
            }
        )
        
        cls.registry.register_agent(
            agent_id="math_computation_agent",
            agent_info={
                "name": "Math Computation Agent",
                "instance": math_agent_mock
            }
        )
        
        cls.registry.register_service(
            service_id="input_processor",
            service_info={
                "name": "Input Processor",
                "instance": input_processor_mock
            }
        )
        
        cls.registry.register_service(
            service_id="content_router",
            service_info={
                "name": "Content Router",
                "instance": content_router_mock
            }
        )
        
        cls.registry.register_service(
            service_id="context_manager",
            service_info={
                "name": "Context Manager",
                "instance": cls.context_manager
            }
        )
    
    @classmethod
    def _create_test_files(cls):
        """Create test files for testing."""
        files = {}
        
        # Create a text file
        text_fd, text_path = tempfile.mkstemp(suffix=".txt")
        os.close(text_fd)
        with open(text_path, 'w') as f:
            f.write("Find the derivative of 2x^2")
        files['text'] = text_path
        
        # Create a mock image file
        img_fd, img_path = tempfile.mkstemp(suffix=".png")
        os.close(img_fd)
        with open(img_path, 'wb') as f:
            # Write a minimal PNG file header
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c````\x00\x00\x00\x05\x00\x01\xa5\xf6E\xbb\x00\x00\x00\x00IEND\xaeB`\x82')
        files['image'] = img_path
        
        return files
    
    def test_text_workflow(self):
        """Test end-to-end workflow with text input."""
        # Create context
        context = self.context_manager.create_context()
        
        # Start workflow
        with open(self.test_files['text'], 'r') as f:
            text_content = f.read()
        
        input_data = {
            "input_type": "text",
            "content": text_content,
            "content_type": "text/plain"
        }
        
        result = self.workflow_manager.start_workflow(
            input_data=input_data,
            context_id=context.context_id
        )
        
        # Verify result
        self.assertIn("workflow_id", result)
        self.assertEqual(result["context_id"], context.context_id)
        
        # Get workflow status
        workflow_id = result["workflow_id"]
        status = self.workflow_manager.get_workflow_status(workflow_id)
        
        # Wait for workflow to complete
        max_wait = 5  # seconds
        start_time = time.time()
        while status["state"] != "completed" and time.time() - start_time < max_wait:
            time.sleep(0.1)
            status = self.workflow_manager.get_workflow_status(workflow_id)
        
        # Verify workflow completed
        self.assertEqual(status["state"], "completed")
        
        # Get workflow result
        workflow_result = self.workflow_manager.get_workflow_result(workflow_id)
        
        # Verify result
        self.assertTrue(workflow_result["success"])
        self.assertIn("result", workflow_result)
        
        # Verify context was updated
        context = self.context_manager.get_context(context.context_id)
        self.assertTrue(len(context.entities) > 0)
    
    def test_image_workflow(self):
        """Test end-to-end workflow with image input."""
        # Create context
        context = self.context_manager.create_context()
        
        # Mock image processing
        self.workflow_manager.input_processor.process_input.return_value = {
            "success": True,
            "input_type": "image",
            "recognized_latex": "2x^2",
            "confidence": 0.9,
            "processing_time_ms": 30
        }
        
        # Start workflow
        input_data = {
            "input_type": "image",
            "content": self.test_files['image'],
            "content_type": "image/png"
        }
        
        result = self.workflow_manager.start_workflow(
            input_data=input_data,
            context_id=context.context_id
        )
        
        # Verify result
        self.assertIn("workflow_id", result)
        self.assertEqual(result["context_id"], context.context_id)
        
        # Get workflow status
        workflow_id = result["workflow_id"]
        status = self.workflow_manager.get_workflow_status(workflow_id)
        
        # Wait for workflow to complete
        max_wait = 5  # seconds
        start_time = time.time()
        while status["state"] != "completed" and time.time() - start_time < max_wait:
            time.sleep(0.1)
            status = self.workflow_manager.get_workflow_status(workflow_id)
        
        # Verify workflow completed
        self.assertEqual(status["state"], "completed")
        
        # Get workflow result
        workflow_result = self.workflow_manager.get_workflow_result(workflow_id)
        
        # Verify result
        self.assertTrue(workflow_result["success"])
        self.assertIn("result", workflow_result)
        
        # Verify context was updated
        context = self.context_manager.get_context(context.context_id)
        self.assertTrue(len(context.entities) > 0)


if __name__ == '__main__':
    unittest.main()

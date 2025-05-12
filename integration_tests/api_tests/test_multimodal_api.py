"""
Integration tests for the multimodal API endpoints.

This module tests the REST and WebSocket API endpoints for multimodal
input processing implemented in Sprint 12.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
from pathlib import Path
import base64

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import FastAPI testing components
from fastapi.testclient import TestClient
from api.rest.server import app


class TestMultimodalAPI(unittest.TestCase):
    """Integration tests for multimodal API endpoints."""
    
    def setUp(self):
        """Set up test environment."""
        self.client = TestClient(app)
        
        # Create test files
        self.test_files = self._create_test_files()
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files
        for file_path in self.test_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def _create_test_files(self):
        """Create test files for integration tests."""
        files = {}
        
        # Create a simple text file
        text_fd, text_path = tempfile.mkstemp(suffix=".txt")
        os.close(text_fd)
        with open(text_path, 'w') as f:
            f.write("Find the derivative of f(x) = x^2 \\sin(x)")
        files['text'] = text_path
        
        # Create a mock image file
        # In a real test, this would be an actual image of handwritten math
        img_fd, img_path = tempfile.mkstemp(suffix=".png")
        os.close(img_fd)
        with open(img_path, 'wb') as f:
            # Write a minimal PNG file header
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c````\x00\x00\x00\x05\x00\x01\xa5\xf6E\xbb\x00\x00\x00\x00IEND\xaeB`\x82')
        files['image'] = img_path
        
        return files
    
    @patch('api.rest.routes.math.process_multimodal_input')
    def test_text_input_endpoint(self, mock_process):
        """Test text input API endpoint."""
        # Mock the processing function
        mock_process.return_value = {
            "success": True,
            "input_type": "text",
            "text_type": "plain",
            "recognized_expression": "f(x) = x^2 \\sin(x)",
            "processing_time_ms": 50
        }
        
        # Test the endpoint
        with open(self.test_files['text'], 'r') as f:
            text_content = f.read()
        
        response = self.client.post(
            "/api/math/input",
            json={
                "input_type": "text",
                "content": text_content
            }
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertEqual(data.get("input_type"), "text")
    
    @patch('api.rest.routes.math.process_multimodal_input')
    def test_image_upload_endpoint(self, mock_process):
        """Test image upload API endpoint."""
        # Mock the processing function
        mock_process.return_value = {
            "success": True,
            "input_type": "image",
            "recognized_latex": "x^2 + y",
            "confidence": 0.95,
            "processing_time_ms": 150
        }
        
        # Test the endpoint
        with open(self.test_files['image'], 'rb') as f:
            image_content = f.read()
        
        # Encode image for JSON transport
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        response = self.client.post(
            "/api/math/input",
            json={
                "input_type": "image",
                "content": base64_image,
                "encoding": "base64",
                "mime_type": "image/png"
            }
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertEqual(data.get("input_type"), "image")
    
    @patch('api.rest.routes.math.process_multimodal_input')
    def test_multipart_input_endpoint(self, mock_process):
        """Test multipart input API endpoint."""
        # Mock the processing function
        mock_process.return_value = {
            "success": True,
            "input_type": "multipart",
            "parts": {
                "text": {
                    "input_type": "text",
                    "text_type": "plain",
                    "text": "Find the derivative"
                },
                "image": {
                    "input_type": "image",
                    "recognized_latex": "x^2 \\sin(x)"
                }
            },
            "processing_time_ms": 200
        }
        
        # Test the endpoint
        with open(self.test_files['text'], 'r') as f:
            text_content = f.read()
        
        with open(self.test_files['image'], 'rb') as f:
            image_content = f.read()
        
        # Encode image for JSON transport
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        response = self.client.post(
            "/api/math/input",
            json={
                "input_type": "multipart",
                "parts": {
                    "text": {
                        "input_type": "text",
                        "content": text_content
                    },
                    "image": {
                        "input_type": "image",
                        "content": base64_image,
                        "encoding": "base64",
                        "mime_type": "image/png"
                    }
                }
            }
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertEqual(data.get("input_type"), "multipart")
    
    @patch('api.rest.routes.math.handle_clarification')
    def test_clarification_endpoint(self, mock_handle):
        """Test clarification API endpoint."""
        # Mock the processing function
        mock_handle.return_value = {
            "success": True,
            "input_type": "image",
            "recognized_latex": "x^2 + y",
            "confidence": 1.0,
            "user_confirmed": True,
            "processing_time_ms": 50
        }
        
        # Test the endpoint
        response = self.client.post(
            "/api/math/clarify",
            json={
                "session_id": "test_session",
                "input_id": "test_input",
                "clarification": {
                    "action": "confirm",
                    "latex": "x^2 + y"
                }
            }
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertTrue(data.get("user_confirmed", False))
    
    @patch('api.rest.routes.math.process_feedback')
    def test_feedback_endpoint(self, mock_process):
        """Test feedback API endpoint."""
        # Mock the processing function
        mock_process.return_value = {
            "success": True,
            "feedback_id": "test_feedback_id",
            "message": "Feedback received"
        }
        
        # Test the endpoint
        response = self.client.post(
            "/api/math/feedback",
            json={
                "type": "correction",
                "user_id": "test_user",
                "session_id": "test_session",
                "input_id": "test_input",
                "original": {
                    "latex": "x^2 + y"
                },
                "correction": {
                    "latex": "x^2 + z"
                }
            }
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertEqual(data.get("feedback_id"), "test_feedback_id")


if __name__ == '__main__':
    unittest.main()

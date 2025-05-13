"""
Unit tests for the unified input pipeline.

This module contains tests for the input processor and content router
implemented for Sprint 12.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from multimodal.unified_pipeline.input_processor import InputProcessor
from multimodal.unified_pipeline.content_router import ContentRouter


class TestInputProcessor(unittest.TestCase):
    """Tests for the unified input processor."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = InputProcessor()
    
    @patch('multimodal.unified_pipeline.input_processor.preprocess_image')
    @patch('multimodal.unified_pipeline.input_processor.detect_symbols')
    @patch('multimodal.unified_pipeline.input_processor.analyze_layout')
    @patch('multimodal.unified_pipeline.input_processor.generate_latex')
    def test_process_image_input(self, mock_generate_latex, mock_analyze_layout, 
                               mock_detect_symbols, mock_preprocess_image):
        """Test processing of image input."""
        # Setup mocks
        mock_preprocess_image.return_value = "preprocessed_image"
        mock_detect_symbols.return_value = [{"text": "x", "position": [10, 10, 20, 20], "confidence": 0.95}]
        mock_analyze_layout.return_value = {"type": "expression"}
        mock_generate_latex.return_value = "x"
        
        # Create a temporary file for testing
        with open("test_image.png", "w") as f:
            f.write("test")
        
        try:
            # Test with file path
            result = self.processor.process_input("test_image.png", "image/png")
            
            # Verify result
            self.assertTrue(result["success"])
            self.assertEqual(result["input_type"], "image")
            self.assertEqual(result["recognized_latex"], "x")
            
            # Verify mock calls
            mock_preprocess_image.assert_called_once()
            mock_detect_symbols.assert_called_once()
            mock_analyze_layout.assert_called_once()
            mock_generate_latex.assert_called_once()
            
        finally:
            # Clean up
            if os.path.exists("test_image.png"):
                os.remove("test_image.png")
    
    def test_process_text_input(self):
        """Test processing of text input."""
        # Test with plain text
        text_input = "Find the derivative of f(x) = x^2 sin(x)"
        result = self.processor.process_input(text_input, "text/plain")
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["input_type"], "text")
        self.assertEqual(result["text_type"], "plain")
        
        # Test with LaTeX
        latex_input = "\\frac{d}{dx}[x^2 \\sin(x)]"
        result = self.processor.process_input(latex_input, "text/x-latex")
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["input_type"], "text")
        self.assertEqual(result["text_type"], "latex")
    
    def test_detect_input_type(self):
        """Test detection of input type."""
        # Test string detection
        self.assertEqual(self.processor._detect_input_type("text"), "text/plain")
        
        # Test bytes detection (simplified)
        pdf_bytes = b'%PDF' + b'x' * 10
        self.assertEqual(self.processor._detect_input_type(pdf_bytes), "application/pdf")
        
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'x' * 10
        self.assertEqual(self.processor._detect_input_type(png_bytes), "image/png")
        
        # Test dictionary detection
        self.assertEqual(self.processor._detect_input_type({"type": "text/html"}), "text/html")
        self.assertEqual(self.processor._detect_input_type({"part1": {}, "part2": {}}), "multipart/mixed")


class TestContentRouter(unittest.TestCase):
    """Tests for the content router."""
    
    def setUp(self):
        """Set up test environment."""
        self.router = ContentRouter()
    
    @patch('multimodal.agent.ocr_agent.OCRAgent.process')
    def test_route_image_content(self, mock_process):
        """Test routing of image content."""
        # Setup mock
        mock_process.return_value = {
            "success": True, 
            "recognized_latex": "x^2"
        }
        
        # Test routing
        image_data = {
            "input_type": "image",
            "image_type": "image/png",
            "recognized_latex": "x^2"
        }
        
        result = self.router.route_content(image_data)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["source_type"], "image")
        self.assertEqual(result["agent_type"], "ocr")
        
        # Verify mock calls
        mock_process.assert_called_once()
    
    @patch('multimodal.agent.advanced_ocr_agent.AdvancedOCRAgent.process')
    def test_route_pdf_content(self, mock_process):
        """Test routing of PDF content."""
        # Setup mock
        mock_process.return_value = {
            "success": True, 
            "recognized_latex": "x^2"
        }
        
        # Test routing
        pdf_data = {
            "input_type": "image",
            "image_type": "application/pdf",
            "recognized_latex": "x^2"
        }
        
        result = self.router.route_content(pdf_data)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["source_type"], "image")
        self.assertEqual(result["agent_type"], "advanced_ocr")
        
        # Verify mock calls
        mock_process.assert_called_once()
    
    def test_route_text_content(self):
        """Test routing of text content."""
        # Test routing
        text_data = {
            "input_type": "text",
            "text_type": "plain",
            "text": "Find the derivative"
        }
        
        result = self.router.route_content(text_data)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["source_type"], "text")
        self.assertEqual(result["agent_type"], "core_llm")


if __name__ == '__main__':
    unittest.main()

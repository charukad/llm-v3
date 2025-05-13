"""
Integration tests for the unified multimodal input processing pipeline.

This module contains end-to-end tests for the Multimodal Input Integration
implemented in Sprint 12.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from multimodal.unified_pipeline.input_processor import InputProcessor
from multimodal.unified_pipeline.content_router import ContentRouter
from multimodal.context.context_manager import ContextManager
from multimodal.context.reference_resolver import ReferenceResolver
from multimodal.interaction.ambiguity_handler import AmbiguityHandler
from multimodal.interaction.feedback_processor import FeedbackProcessor


class TestMultimodalIntegration(unittest.TestCase):
    """Integration tests for multimodal input processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_processor = InputProcessor()
        self.content_router = ContentRouter()
        self.context_manager = ContextManager()
        self.reference_resolver = ReferenceResolver()
        self.ambiguity_handler = AmbiguityHandler()
        self.feedback_processor = FeedbackProcessor()
        
        # Create a test context
        self.test_context = self.context_manager.create_context()
        
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
    
    @patch('multimodal.ocr.advanced_symbol_detector.detect_symbols')
    @patch('multimodal.structure.layout_analyzer.analyze_layout')
    @patch('multimodal.latex_generator.latex_generator.generate_latex')
    def test_end_to_end_image_processing(self, mock_generate_latex, mock_analyze_layout, mock_detect_symbols):
        """Test end-to-end processing of an image input."""
        # Set up mocks
        mock_detect_symbols.return_value = [
            {"text": "x", "position": [10, 10, 20, 20], "confidence": 0.95},
            {"text": "2", "position": [20, 5, 25, 15], "confidence": 0.90},
            {"text": "+", "position": [30, 10, 40, 20], "confidence": 0.98},
            {"text": "y", "position": [50, 10, 60, 20], "confidence": 0.92}
        ]
        
        mock_analyze_layout.return_value = {
            "type": "expression",
            "elements": [
                {"type": "variable", "value": "x"},
                {"type": "superscript", "value": "2"},
                {"type": "operator", "value": "+"},
                {"type": "variable", "value": "y"}
            ]
        }
        
        mock_generate_latex.return_value = "x^2 + y"
        
        # Process the test image
        processed_input = self.input_processor.process_input(self.test_files['image'])
        
        # Verify basic processing
        self.assertTrue(processed_input.get("success", False))
        self.assertEqual(processed_input.get("input_type"), "image")
        
        # Check OCR processing
        self.assertEqual(processed_input.get("recognized_latex"), "x^2 + y")
        
        # Check for ambiguities
        ambiguities = self.ambiguity_handler.detect_ambiguities(processed_input)
        self.assertFalse(ambiguities.get("has_ambiguities", True))
        
        # Route content
        routing_result = self.content_router.route_content(processed_input)
        self.assertTrue(routing_result.get("success", False))
        
        # Add to context
        entity_id = self.context_manager.add_entity_to_context(
            self.test_context.context_id,
            {
                "type": "expression",
                "latex": processed_input.get("recognized_latex"),
                "source": "image"
            },
            "image"
        )
        
        self.assertIsNotNone(entity_id)
        
        # Verify entity was added to context
        entity = self.test_context.get_entity(entity_id)
        self.assertIsNotNone(entity)
        self.assertEqual(entity.get("latex"), "x^2 + y")
    
    def test_end_to_end_text_processing(self):
        """Test end-to-end processing of a text input."""
        # Process the test text file
        with open(self.test_files['text'], 'r') as f:
            text_content = f.read()
        
        processed_input = self.input_processor.process_input(text_content)
        
        # Verify basic processing
        self.assertTrue(processed_input.get("success", False))
        self.assertEqual(processed_input.get("input_type"), "text")
        
        # Route content
        routing_result = self.content_router.route_content(processed_input)
        self.assertTrue(routing_result.get("success", False))
        
        # Add to context
        entity_id = self.context_manager.add_entity_to_context(
            self.test_context.context_id,
            {
                "type": "query",
                "text": text_content,
                "source": "text"
            },
            "text"
        )
        
        self.assertIsNotNone(entity_id)
        
        # Verify entity was added to context
        entity = self.test_context.get_entity(entity_id)
        self.assertIsNotNone(entity)
        self.assertTrue("Find the derivative" in entity.get("text", ""))
    
    @patch('multimodal.ocr.advanced_symbol_detector.detect_symbols')
    @patch('multimodal.structure.layout_analyzer.analyze_layout')
    @patch('multimodal.latex_generator.latex_generator.generate_latex')
    def test_ambiguity_handling(self, mock_generate_latex, mock_analyze_layout, mock_detect_symbols):
        """Test handling of ambiguous input."""
        # Set up mocks for an ambiguous image
        mock_detect_symbols.return_value = [
            {"text": "x", "position": [10, 10, 20, 20], "confidence": 0.95},
            {"text": "?", "position": [20, 5, 25, 15], "confidence": 0.4},  # Ambiguous symbol
            {"text": "+", "position": [30, 10, 40, 20], "confidence": 0.98},
            {"text": "y", "position": [50, 10, 60, 20], "confidence": 0.92}
        ]
        
        mock_analyze_layout.return_value = {
            "type": "expression",
            "elements": [
                {"type": "variable", "value": "x"},
                {"type": "superscript", "value": "?"},
                {"type": "operator", "value": "+"},
                {"type": "variable", "value": "y"}
            ]
        }
        
        mock_generate_latex.return_value = "x^? + y"
        
        # Process the test image
        processed_input = self.input_processor.process_input(self.test_files['image'])
        processed_input["confidence"] = 0.7  # Force low confidence for testing
        
        # Detect ambiguities
        ambiguities = self.ambiguity_handler.detect_ambiguities(processed_input)
        self.assertTrue(ambiguities.get("has_ambiguities", False))
        
        # Generate clarification request
        clarification_request = self.ambiguity_handler.generate_clarification_request(ambiguities)
        self.assertTrue(clarification_request.get("needs_clarification", False))
        
        # Simulate user clarification
        user_clarification = {
            "action": "edit",
            "latex": "x^2 + y"  # User corrected the ambiguous part
        }
        
        # Process clarification
        updated_input = self.ambiguity_handler.process_clarification(processed_input, user_clarification)
        
        # Verify clarification was applied
        self.assertEqual(updated_input.get("recognized_latex"), "x^2 + y")
        self.assertTrue(updated_input.get("user_edited", False))
    
    def test_cross_modal_context(self):
        """Test cross-modal context management."""
        # Add text entity to context
        text_entity_id = self.context_manager.add_entity_to_context(
            self.test_context.context_id,
            {
                "type": "expression",
                "latex": "f(x) = x^2 \\sin(x)",
                "source": "text"
            },
            "text"
        )
        
        # Add image entity to context
        image_entity_id = self.context_manager.add_entity_to_context(
            self.test_context.context_id,
            {
                "type": "expression",
                "latex": "x^2 \\sin(x)",
                "source": "image"
            },
            "image"
        )
        
        # Create a reference between them
        reference = self.context_manager.add_reference_to_context(
            self.test_context.context_id,
            text_entity_id,
            image_entity_id,
            "represents"
        )
        
        self.assertIsNotNone(reference)
        
        # Retrieve the cross-references
        text_refs = self.test_context.find_references(text_entity_id)
        image_refs = self.test_context.find_references(image_entity_id)
        
        self.assertEqual(len(text_refs), 1)
        self.assertEqual(len(image_refs), 1)
        self.assertEqual(text_refs[0]["target_id"], image_entity_id)
        self.assertEqual(image_refs[0]["source_id"], text_entity_id)
    
    def test_user_feedback_processing(self):
        """Test processing of user feedback."""
        feedback_data = {
            "type": "correction",
            "user_id": "test_user",
            "original": {
                "latex": "x^2 + y"
            },
            "correction": {
                "latex": "x^2 + z"
            },
            "comment": "Variable should be z, not y"
        }
        
        result = self.feedback_processor.process_feedback(feedback_data)
        self.assertTrue(result.get("success", False))
        self.assertIsNotNone(result.get("feedback_id"))


if __name__ == '__main__':
    unittest.main()

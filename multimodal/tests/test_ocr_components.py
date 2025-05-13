"""
Unit Tests for OCR Components

This module contains unit tests for the OCR components of the system,
including image preprocessing, symbol detection, and LaTeX generation.
"""

import unittest
import os
import cv2
import numpy as np
from multimodal.image_processing.preprocessor import ImagePreprocessor
from multimodal.ocr.symbol_detector import MathSymbolDetector
from multimodal.latex_generator.latex_generator import LaTeXGenerator

class OCRComponentsTest(unittest.TestCase):
    """Tests for OCR components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image with "1+1=2"
        self.test_image = np.ones((100, 300), dtype=np.uint8) * 255
        cv2.putText(self.test_image, "1+1=2", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.detector = MathSymbolDetector()
        self.latex_generator = LaTeXGenerator()
    
    def test_image_preprocessing(self):
        """Test the image preprocessing component."""
        # Process the test image
        result = self.preprocessor.preprocess(self.test_image)
        
        # Check the structure of the result
        self.assertIn("processed_image", result)
        self.assertIn("metadata", result)
        
        # Check if the processed image is valid
        processed_image = result["processed_image"]
        self.assertIsInstance(processed_image, np.ndarray)
        
        # Check metadata
        metadata = result["metadata"]
        self.assertIn("original_shape", metadata)
        self.assertIn("preprocessing_steps", metadata)
        self.assertIsInstance(metadata["preprocessing_steps"], list)
    
    def test_symbol_detection(self):
        """Test the symbol detection component."""
        # Preprocess the image first
        preprocessing_result = self.preprocessor.preprocess(self.test_image)
        processed_image = preprocessing_result["processed_image"]
        
        # Detect symbols
        result = self.detector.detect_symbols(processed_image)
        
        # Check the structure of the result
        self.assertIn("symbols", result)
        self.assertIn("count", result)
        self.assertIn("metadata", result)
        
        # This part of the test is more complicated because it depends on
        # the actual symbol detection, which may not be reliable with our
        # simple test image. In a real test, you'd use a known test image with
        # ground truth annotations.
        
        # For now, we just check if the symbols list is present
        symbols = result["symbols"]
        self.assertIsInstance(symbols, list)
    
    def test_layout_analysis(self):
        """Test the layout analysis component."""
        # Preprocess the image first
        preprocessing_result = self.preprocessor.preprocess(self.test_image)
        processed_image = preprocessing_result["processed_image"]
        
        # Analyze layout
        result = self.detector.analyze_layout(processed_image)
        
        # Check the structure of the result
        self.assertIn("symbols", result)
        self.assertIn("lines", result)
        self.assertIn("structures", result)
        self.assertIn("count", result)
        self.assertIn("metadata", result)
        
        # Check lines
        lines = result["lines"]
        self.assertIsInstance(lines, list)
        
        # Check structures
        structures = result["structures"]
        self.assertIsInstance(structures, list)
    
    def test_latex_generation(self):
        """Test the LaTeX generation component."""
        # Create some mock layout analysis results
        mock_analysis = {
            "symbols": [
                {
                    "text": "1",
                    "math_symbol": "1",
                    "confidence": 0.9,
                    "position": [[50, 50], [60, 50], [60, 60], [50, 60]]
                },
                {
                    "text": "+",
                    "math_symbol": "+",
                    "confidence": 0.8,
                    "position": [[70, 50], [80, 50], [80, 60], [70, 60]]
                },
                {
                    "text": "1",
                    "math_symbol": "1",
                    "confidence": 0.9,
                    "position": [[90, 50], [100, 50], [100, 60], [90, 60]]
                },
                {
                    "text": "=",
                    "math_symbol": "=",
                    "confidence": 0.8,
                    "position": [[110, 50], [120, 50], [120, 60], [110, 60]]
                },
                {
                    "text": "2",
                    "math_symbol": "2",
                    "confidence": 0.9,
                    "position": [[130, 50], [140, 50], [140, 60], [130, 60]]
                }
            ],
            "lines": [
                {
                    "symbols": [
                        {
                            "text": "1",
                            "math_symbol": "1",
                            "confidence": 0.9,
                            "position": [[50, 50], [60, 50], [60, 60], [50, 60]]
                        },
                        {
                            "text": "+",
                            "math_symbol": "+",
                            "confidence": 0.8,
                            "position": [[70, 50], [80, 50], [80, 60], [70, 60]]
                        },
                        {
                            "text": "1",
                            "math_symbol": "1",
                            "confidence": 0.9,
                            "position": [[90, 50], [100, 50], [100, 60], [90, 60]]
                        },
                        {
                            "text": "=",
                            "math_symbol": "=",
                            "confidence": 0.8,
                            "position": [[110, 50], [120, 50], [120, 60], [110, 60]]
                        },
                        {
                            "text": "2",
                            "math_symbol": "2",
                            "confidence": 0.9,
                            "position": [[130, 50], [140, 50], [140, 60], [130, 60]]
                        }
                    ],
                    "y_position": 55
                }
            ],
            "structures": []
        }
        
        # Generate LaTeX
        result = self.latex_generator.generate_latex(mock_analysis)
        
        # Check the structure of the result
        self.assertIn("latex", result)
        self.assertIn("display_latex", result)
        self.assertIn("full_latex", result)
        self.assertIn("full_display_latex", result)
        self.assertIn("metadata", result)
        
        # Check if the LaTeX is as expected
        latex = result["latex"]
        self.assertEqual(latex, "1+1=2")
        
        # Check the display LaTeX
        display_latex = result["display_latex"]
        self.assertEqual(display_latex, "\\displaystyle {1+1=2}")

if __name__ == "__main__":
    unittest.main()

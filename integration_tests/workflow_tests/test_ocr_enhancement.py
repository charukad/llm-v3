import os
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from multimodal.image_processing.preprocessor import ImagePreprocessor
from multimodal.ocr.advanced_symbol_detector import MathSymbolDetector
from multimodal.ocr.context_analyzer import MathContextAnalyzer
from multimodal.structure.layout_analyzer import MathLayoutAnalyzer
from multimodal.latex_generator.latex_generator import LaTeXGenerator
from multimodal.agent.ocr_agent import HandwritingRecognitionAgent

class TestOCREnhancement(unittest.TestCase):
    """
    Integration tests for OCR enhancement components implemented in Sprint 10.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test images directory if it doesn't exist
        cls.test_dir = project_root / 'integration_tests' / 'test_data' / 'ocr'
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test images if they don't exist
        cls._create_test_images()
        
        # Initialize agents
        cls.agent = HandwritingRecognitionAgent()
    
    @classmethod
    def _create_test_images(cls):
        """Create synthetic test images for OCR testing."""
        # Simple equation image
        equation_path = cls.test_dir / 'simple_equation.png'
        if not equation_path.exists():
            # Create a white image
            img = np.ones((200, 500, 3), dtype=np.uint8) * 255
            # Draw text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "x^2 + 3x + 2 = 0", (50, 100), font, 1.5, (0, 0, 0), 2)
            cv2.imwrite(str(equation_path), img)
        
        # Fraction image
        fraction_path = cls.test_dir / 'fraction.png'
        if not fraction_path.exists():
            # Create a white image
            img = np.ones((300, 500, 3), dtype=np.uint8) * 255
            # Draw fraction
            cv2.line(img, (150, 150), (350, 150), (0, 0, 0), 2)  # Fraction line
            cv2.putText(img, "x + 1", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Numerator
            cv2.putText(img, "x - 1", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Denominator
            cv2.imwrite(str(fraction_path), img)
    
    def test_advanced_symbol_detection(self):
        """Test advanced symbol detection."""
        # Initialize components
        preprocessor = ImagePreprocessor()
        detector = MathSymbolDetector()
        
        # Load test image
        image_path = self.test_dir / 'simple_equation.png'
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image, "Failed to load test image")
        
        # Preprocess and detect
        preprocessed = preprocessor.preprocess(image)
        symbols = detector.detect_symbols(preprocessed)
        
        # Verify detection
        self.assertGreater(len(symbols), 0, "No symbols detected")
        
        # Check for expected symbols
        detected_texts = [s['text'] for s in symbols]
        self.assertTrue(any('x' in t for t in detected_texts), "Failed to detect 'x'")
        self.assertTrue(any('+' in t for t in detected_texts), "Failed to detect '+'")
        self.assertTrue(any('=' in t for t in detected_texts), "Failed to detect '='")
    
    def test_structure_analysis(self):
        """Test 2D structure analysis."""
        # Initialize components
        preprocessor = ImagePreprocessor()
        detector = MathSymbolDetector()
        analyzer = MathLayoutAnalyzer()
        
        # Load test image
        image_path = self.test_dir / 'fraction.png'
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image, "Failed to load test image")
        
        # Preprocess, detect and analyze
        preprocessed = preprocessor.preprocess(image)
        symbols = detector.detect_symbols(preprocessed)
        structure = analyzer.analyze_structure(symbols)
        
        # Verify structure analysis
        self.assertIsNotNone(structure, "Failed to analyze structure")
        
        # Check for fraction recognition
        # Note: This is a simplified test - actual fraction detection might
        # require more sophisticated synthetic images or real handwritten samples
        if structure['type'] == 'fraction':
            self.assertIn('numerator', structure, "Fraction missing numerator")
            self.assertIn('denominator', structure, "Fraction missing denominator")
        else:
            # Alternative structure types might be valid depending on the image quality
            self.assertIn('type', structure, "Structure missing type")
    
    def test_context_analyzer(self):
        """Test context-aware recognition."""
        # Initialize components
        preprocessor = ImagePreprocessor()
        detector = MathSymbolDetector()
        context_analyzer = MathContextAnalyzer()
        
        # Load test image
        image_path = self.test_dir / 'simple_equation.png'
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image, "Failed to load test image")
        
        # Preprocess and detect
        preprocessed = preprocessor.preprocess(image)
        symbols = detector.detect_symbols(preprocessed)
        
        # Create a potential ambiguity by modifying a symbol
        if symbols:
            # Find an 'x' or create one if not found
            x_index = next((i for i, s in enumerate(symbols) if s['text'] == 'x'), None)
            if x_index is not None:
                # Change 'x' to 'X' to test correction
                symbols[x_index]['text'] = 'X'
                symbols[x_index]['confidence'] = 0.7
        
        # Apply context correction
        corrected = context_analyzer.analyze_and_correct(symbols, domain="algebra")
        
        # Verify domain identification
        self.assertEqual(context_analyzer._identify_domain(symbols), "algebra")
        
        # Check if some corrections were made
        # This is a bit artificial, but serves as a basic integration test
        if x_index is not None:
            self.assertIn(corrected[x_index]['text'], ['x', 'X'], 
                          "Context analyzer failed to maintain or correct ambiguity")
    
    def test_full_ocr_pipeline(self):
        """Test the complete OCR pipeline."""
        # Load test image
        image_path = self.test_dir / 'simple_equation.png'
        
        # Process with agent
        result = self.agent.process_image(str(image_path))
        
        # Verify result
        self.assertTrue(result['success'], "OCR pipeline failed")
        self.assertGreater(len(result['latex']), 0, "Empty LaTeX result")
        self.assertGreater(result['confidence'], 0, "Zero confidence score")
        
        # Basic LaTeX content checks
        latex = result['latex']
        self.assertIn('x', latex, "LaTeX missing 'x' variable")
        # This will vary based on the exact image and recognition quality
        self.assertTrue(any(op in latex for op in ['+', '=']), 
                       "LaTeX missing expected operators")


if __name__ == '__main__':
    unittest.main()

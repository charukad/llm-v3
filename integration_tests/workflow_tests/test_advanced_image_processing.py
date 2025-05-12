import os
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
import tempfile

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from multimodal.image_processing.format_handler import FormatHandler
from multimodal.image_processing.diagram_detector import DiagramDetector
from multimodal.image_processing.coordinate_detector import CoordinateSystemDetector
from multimodal.agent.advanced_ocr_agent import AdvancedOCRAgent

class TestAdvancedImageProcessing(unittest.TestCase):
    """
    Integration tests for advanced image processing capabilities implemented in Sprint 11.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test directory and images
        cls.test_dir = tempfile.mkdtemp()
        cls.test_images = cls._create_test_images(cls.test_dir)
        
        # Initialize components
        cls.format_handler = FormatHandler()
        cls.diagram_detector = DiagramDetector()
        cls.coordinate_detector = CoordinateSystemDetector()
        cls.ocr_agent = AdvancedOCRAgent()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up test images
        for image_path in cls.test_images:
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # Remove test directory
        if os.path.exists(cls.test_dir):
            os.rmdir(cls.test_dir)
    
    @classmethod
    def _create_test_images(cls, output_dir):
        """Create test images for testing."""
        image_paths = []
        
        # Create a coordinate system image
        coord_path = os.path.join(output_dir, "coordinate_system.png")
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw axes
        cv2.line(img, (50, 200), (350, 200), (0, 0, 0), 2)  # x-axis
        cv2.line(img, (200, 350), (200, 50), (0, 0, 0), 2)  # y-axis
        
        # Draw grid lines
        for i in range(100, 350, 50):
            cv2.line(img, (i, 197), (i, 203), (0, 0, 0), 1)  # x-axis ticks
            cv2.line(img, (197, i), (203, i), (0, 0, 0), 1)  # y-axis ticks
        
        # Draw a function curve
        pts = []
        for x in range(50, 350):
            y = 200 - int(100 * np.sin((x - 50) / 50))
            pts.append((x, y))
        
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (255, 0, 0), 2)
        
        cv2.imwrite(coord_path, img)
        image_paths.append(coord_path)
        
        # Create a geometric diagram
        diagram_path = os.path.join(output_dir, "diagram.png")
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw a triangle
        cv2.line(img, (100, 300), (200, 100), (0, 0, 0), 2)
        cv2.line(img, (200, 100), (300, 300), (0, 0, 0), 2)
        cv2.line(img, (300, 300), (100, 300), (0, 0, 0), 2)
        
        # Draw a circle
        cv2.circle(img, (200, 200), 50, (0, 0, 0), 2)
        
        cv2.imwrite(diagram_path, img)
        image_paths.append(diagram_path)
        
        # Create a multi-region image
        multi_path = os.path.join(output_dir, "multi_region.png")
        img = np.ones((600, 400, 3), dtype=np.uint8) * 255
        
        # Draw an equation in the first region
        cv2.putText(img, "x^2 + 2x = 5", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Draw a small coordinate system in the second region
        cv2.line(img, (50, 300), (350, 300), (0, 0, 0), 2)  # x-axis
        cv2.line(img, (200, 400), (200, 200), (0, 0, 0), 2)  # y-axis
        
        cv2.imwrite(multi_path, img)
        image_paths.append(multi_path)
        
        return image_paths
    
    def test_format_handler(self):
        """Test the format handler functionality."""
        for image_path in self.test_images:
            # Test file loading
            result = self.format_handler.load_file(image_path)
            
            self.assertTrue(result['success'], f"Failed to load file: {image_path}")
            self.assertEqual(len(result['images']), 1, f"Expected 1 image in the file")
            
            # Test image splitting
            image = result['images'][0]
            regions = self.format_handler.split_image(image)
            
            self.assertIsInstance(regions, list, "Regions should be a list")
            self.assertGreaterEqual(len(regions), 1, "Should detect at least one region")
            
            # Verify region structure
            for region in regions:
                self.assertIn('image', region, "Region should contain 'image' key")
                self.assertIn('x', region, "Region should contain 'x' key")
                self.assertIn('y', region, "Region should contain 'y' key")
                self.assertIn('width', region, "Region should contain 'width' key")
                self.assertIn('height', region, "Region should contain 'height' key")
    
    def test_diagram_detector(self):
        """Test the diagram detector functionality."""
        diagram_path = os.path.join(self.test_dir, "diagram.png")
        
        # Check if the file exists
        self.assertTrue(os.path.exists(diagram_path), f"Test image not found: {diagram_path}")
        
        # Load image
        image = cv2.imread(diagram_path)
        self.assertIsNotNone(image, f"Failed to load image: {diagram_path}")
        
        # Test diagram type detection
        detection_result = self.diagram_detector.detect_diagram_type(image)
        
        self.assertIsInstance(detection_result, dict, "Detection result should be a dictionary")
        self.assertIn('type', detection_result, "Result should contain 'type' key")
        self.assertIn('confidence', detection_result, "Result should contain 'confidence' key")
        
        # The test image is a geometric diagram, so it should be detected as such
        self.assertGreaterEqual(detection_result['confidence'], 0.5, 
                              "Confidence should be at least 0.5")
        
        # Test diagram analysis
        analysis_result = self.diagram_detector.analyze_diagram(image)
        
        self.assertIsInstance(analysis_result, dict, "Analysis result should be a dictionary")
        
        # Test LaTeX extraction
        latex = self.diagram_detector.extract_latex(analysis_result)
        self.assertIsInstance(latex, str, "LaTeX should be a string")
        self.assertGreater(len(latex), 0, "LaTeX should not be empty")
    
    def test_coordinate_detector(self):
        """Test the coordinate system detector functionality."""
        coord_path = os.path.join(self.test_dir, "coordinate_system.png")
        
        # Check if the file exists
        self.assertTrue(os.path.exists(coord_path), f"Test image not found: {coord_path}")
        
        # Load image
        image = cv2.imread(coord_path)
        self.assertIsNotNone(image, f"Failed to load image: {coord_path}")
        
        # Test coordinate system detection
        detection_result = self.coordinate_detector.detect_coordinate_system(image)
        
        self.assertIsInstance(detection_result, dict, "Detection result should be a dictionary")
        self.assertIn('type', detection_result, "Result should contain 'type' key")
        self.assertIn('confidence', detection_result, "Result should contain 'confidence' key")
        self.assertIn('details', detection_result, "Result should contain 'details' key")
        
        # The test image is a Cartesian coordinate system, so it should be detected
        self.assertEqual(detection_result['type'], 'cartesian_2d', 
                       "Should detect a 2D Cartesian coordinate system")
        self.assertGreaterEqual(detection_result['confidence'], 0.7, 
                              "Confidence should be at least 0.7")
        
        # Test function extraction
        functions = self.coordinate_detector.extract_functions(image, detection_result)
        
        self.assertIsInstance(functions, list, "Functions should be a list")
        self.assertGreaterEqual(len(functions), 1, "Should detect at least one function")
    
    def test_advanced_ocr_agent(self):
        """Test the advanced OCR agent functionality."""
        for image_path in self.test_images:
            # Test file processing
            result = self.ocr_agent.process_file(image_path)
            
            self.assertTrue(result['success'], f"Processing failed for {image_path}")
            self.assertIn('type', result, "Result should contain 'type' key")
            self.assertIn('latex', result, "Result should contain 'latex' key")
            
            # LaTeX should be non-empty
            self.assertGreater(len(result['latex']), 0, "LaTeX should not be empty")

if __name__ == '__main__':
    unittest.main()

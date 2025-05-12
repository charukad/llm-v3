#!/usr/bin/env python3
"""
Test script for advanced image processing capabilities implemented in Sprint 11.
"""

import os
import sys
import argparse
import logging
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our components
from multimodal.image_processing.format_handler import FormatHandler
from multimodal.image_processing.diagram_detector import DiagramDetector
from multimodal.image_processing.coordinate_detector import CoordinateSystemDetector
from multimodal.agent.advanced_ocr_agent import AdvancedOCRAgent

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def create_test_images(output_dir):
    """Create test images for different types of mathematical content."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple coordinate system
    coordinate_path = os.path.join(output_dir, 'coordinate_system.png')
    if not os.path.exists(coordinate_path):
        # Create a white image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        # Draw coordinate axes
        cv2.line(img, (50, 250), (450, 250), (0, 0, 0), 2)  # x-axis
        cv2.line(img, (250, 450), (250, 50), (0, 0, 0), 2)  # y-axis
        
        # Draw grid lines
        for i in range(100, 450, 50):
            cv2.line(img, (i, 245), (i, 255), (0, 0, 0), 1)  # x-axis ticks
            cv2.line(img, (245, i), (255, i), (0, 0, 0), 1)  # y-axis ticks
        
        # Draw a function curve
        pts = []
        for x in range(50, 450):
            y = 250 - int(100 * np.sin((x - 50) / 100))
            pts.append((x, y))
        
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (255, 0, 0), 2)
        
        cv2.imwrite(coordinate_path, img)
        print(f"Created coordinate system test image: {coordinate_path}")
    
    # Create a simple diagram (geometric)
    diagram_path = os.path.join(output_dir, 'geometric_diagram.png')
    if not os.path.exists(diagram_path):
        # Create a white image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        # Draw a triangle
        cv2.line(img, (100, 400), (250, 100), (0, 0, 0), 2)
        cv2.line(img, (250, 100), (400, 400), (0, 0, 0), 2)
        cv2.line(img, (400, 400), (100, 400), (0, 0, 0), 2)
        
        # Draw a circle
        cv2.circle(img, (250, 250), 50, (0, 0, 0), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "A", (90, 420), font, 1, (0, 0, 0), 2)
        cv2.putText(img, "B", (240, 90), font, 1, (0, 0, 0), 2)
        cv2.putText(img, "C", (410, 420), font, 1, (0, 0, 0), 2)
        cv2.putText(img, "O", (260, 260), font, 1, (0, 0, 0), 2)
        
        cv2.imwrite(diagram_path, img)
        print(f"Created geometric diagram test image: {diagram_path}")
    
    # Create a multi-region image
    multi_region_path = os.path.join(output_dir, 'multi_region.png')
    if not os.path.exists(multi_region_path):
        # Create a white image
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Draw an equation in the first region
        cv2.putText(img, "f(x) = x^2 + 2x + 1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Draw a coordinate system in the second region
        cv2.line(img, (50, 300), (550, 300), (0, 0, 0), 2)  # x-axis
        cv2.line(img, (300, 150), (300, 450), (0, 0, 0), 2)  # y-axis
        
        # Draw a curve
        pts = []
        for x in range(50, 550):
            y = 300 - int(0.001 * (x - 300)**2)
            pts.append((x, y))
        
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (255, 0, 0), 2)
        
        # Draw another equation in the third region
        cv2.putText(img, "int_0^1 x^2 dx = 1/3", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imwrite(multi_region_path, img)
        print(f"Created multi-region test image: {multi_region_path}")
    
    return [coordinate_path, diagram_path, multi_region_path]

def test_format_handler(file_path):
    """Test the format handler with different file types."""
    format_handler = FormatHandler()
    
    print(f"\nTesting format handler on: {file_path}")
    start_time = time.time()
    
    # Load file
    result = format_handler.load_file(file_path)
    
    elapsed = time.time() - start_time
    print(f"Format handler processed file in {elapsed:.3f} seconds")
    
    if result['success']:
        print(f"Detected format: {result['format']}")
        print(f"Found {len(result['images'])} images")
        
        # If it's a multi-page document, process page by page
        if len(result['images']) > 1:
            print("Multi-page document detected, processing regions by page...")
            
            for i, image in enumerate(result['images']):
                print(f"\nProcessing page {i+1}:")
                regions = format_handler.split_image(image)
                print(f"Found {len(regions)} regions on page {i+1}")
                
                # Display information about each region
                for j, region in enumerate(regions):
                    print(f"  Region {j+1}: {region['width']}x{region['height']} at ({region['x']}, {region['y']})")
        else:
            # Single page - process regions
            image = result['images'][0]
            regions = format_handler.split_image(image)
            print(f"Found {len(regions)} regions")
            
            for i, region in enumerate(regions):
                print(f"  Region {i+1}: {region['width']}x{region['height']} at ({region['x']}, {region['y']})")
    else:
        print(f"Error: {result['error']}")
    
    return result

def test_diagram_detector(image_path):
    """Test the diagram detector on an image."""
    diagram_detector = DiagramDetector()
    
    print(f"\nTesting diagram detector on: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Detect diagram type
    start_time = time.time()
    detection_result = diagram_detector.detect_diagram_type(image)
    elapsed = time.time() - start_time
    
    print(f"Diagram type detection completed in {elapsed:.3f} seconds")
    print(f"Detected diagram type: {detection_result['type']} with confidence {detection_result['confidence']:.2f}")
    
    # Analyze diagram
    start_time = time.time()
    analysis_result = diagram_detector.analyze_diagram(image)
    elapsed = time.time() - start_time
    
    print(f"Diagram analysis completed in {elapsed:.3f} seconds")
    
    # Generate LaTeX
    latex = diagram_detector.extract_latex(analysis_result)
    print(f"Generated LaTeX representation:")
    print(latex)
    
    return analysis_result

def test_coordinate_detector(image_path):
    """Test the coordinate system detector on an image."""
    coordinate_detector = CoordinateSystemDetector()
    
    print(f"\nTesting coordinate system detector on: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Detect coordinate system
    start_time = time.time()
    detection_result = coordinate_detector.detect_coordinate_system(image)
    elapsed = time.time() - start_time
    
    print(f"Coordinate system detection completed in {elapsed:.3f} seconds")
    print(f"Detected system type: {detection_result['type']} with confidence {detection_result['confidence']:.2f}")
    
    # Extract functions if it's a coordinate system
    if detection_result['confidence'] > 0.5:
        functions = coordinate_detector.extract_functions(image, detection_result)
        print(f"Extracted {len(functions)} functions from the coordinate system")
        
        for i, function in enumerate(functions):
            print(f"  Function {i+1} type: {function.get('type', 'unknown')}")
    
    return detection_result

def test_advanced_ocr_agent(file_path):
    """Test the advanced OCR agent on a file."""
    agent = AdvancedOCRAgent()
    
    print(f"\nTesting advanced OCR agent on: {file_path}")
    
    # Process file
    start_time = time.time()
    result = agent.process_file(file_path)
    elapsed = time.time() - start_time
    
    print(f"Advanced OCR processing completed in {elapsed:.3f} seconds")
    
    if result['success']:
        print(f"Processing succeeded")
        print(f"Content type: {result.get('type', 'unknown')}")
        
        # Show additional information based on content type
        if result.get('type') == 'diagram':
            print(f"Diagram type: {result.get('diagram_type', 'unknown')}")
        elif result.get('type') == 'coordinate_system':
            print(f"System type: {result.get('system_type', 'unknown')}")
            print(f"Found {len(result.get('functions', []))} functions")
        elif result.get('type') == 'multi_region':
            print(f"Found {result.get('region_count', 0)} regions")
        elif result.get('type') == 'math_expression':
            print(f"Domain: {result.get('domain', 'unknown')}")
            print(f"Symbol count: {result.get('symbols_count', 0)}")
        
        # Show LaTeX
        print("\nGenerated LaTeX:")
        latex = result.get('latex', '')
        if len(latex) > 500:
            print(latex[:500] + "...")
        else:
            print(latex)
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test advanced image processing components")
    parser.add_argument("--file", help="Path to a file to process")
    parser.add_argument("--component", choices=["format", "diagram", "coordinate", "agent"],
                        help="Test a specific component")
    parser.add_argument("--create-test-images", action="store_true",
                        help="Create test images for testing")
    parser.add_argument("--output-dir", default="test_images",
                        help="Directory for test images")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create test images if requested
    if args.create_test_images:
        test_images = create_test_images(args.output_dir)
        if not args.file and test_images:
            args.file = test_images[0]  # Use the first test image
    
    # Check if file exists
    if args.file and not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1
    
    # Test specific component
    if args.component and args.file:
        if args.component == "format":
            test_format_handler(args.file)
        elif args.component == "diagram":
            test_diagram_detector(args.file)
        elif args.component == "coordinate":
            test_coordinate_detector(args.file)
        elif args.component == "agent":
            test_advanced_ocr_agent(args.file)
    elif args.file:
        # Test all components
        file_info = test_format_handler(args.file)
        
        if file_info['success'] and file_info['images']:
            # Use the first image for component testing
            image_path = args.file
            if len(file_info['images']) > 0:
                # Save the first image temporarily for component testing
                first_image = file_info['images'][0]
                temp_image_path = os.path.join(os.path.dirname(args.file), 
                                            f"temp_test_image_{int(time.time())}.png")
                cv2.imwrite(temp_image_path, first_image)
                image_path = temp_image_path
                print(f"\nSaved temporary image for component testing: {temp_image_path}")
            
            try:
                # Test diagram detector on the image
                test_diagram_detector(image_path)
                
                # Test coordinate system detector on the image
                test_coordinate_detector(image_path)
                
                # Test the advanced OCR agent on the original file
                test_advanced_ocr_agent(args.file)
                
                # Clean up temp file if we created one
                if image_path != args.file and os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"\nRemoved temporary file: {image_path}")
            except Exception as e:
                print(f"Error during testing: {str(e)}")
                # Clean up temp file even if there was an error
                if image_path != args.file and os.path.exists(image_path):
                    os.remove(image_path)
    else:
        # No file specified
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

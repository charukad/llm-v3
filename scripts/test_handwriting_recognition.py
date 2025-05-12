#!/usr/bin/env python3
"""
Test Script for Handwriting Recognition

This script demonstrates the handwriting recognition capabilities for mathematical
notation, showing how the system can process images of handwritten equations and
convert them to LaTeX representation.
"""

import sys
import os
import json
import argparse
from datetime import datetime
import cv2
import matplotlib.pyplot as plt

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multimodal.agent.ocr_agent import HandwritingRecognitionAgent, process_handwritten_image
from multimodal.image_processing.preprocessor import ImagePreprocessor
from multimodal.ocr.symbol_detector import MathSymbolDetector
from multimodal.latex_generator.latex_generator import LaTeXGenerator

def print_separator(text=""):
    """Print a separator line with optional text."""
    width = 80
    if text:
        text = f" {text} "
        padding = (width - len(text)) // 2
        print("=" * padding + text + "=" * (width - padding - len(text)))
    else:
        print("=" * width)

def print_json(data):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def visualize_results(image_path, results):
    """Visualize the detection results on the image."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image from {image_path}")
        return
    
    # Convert to RGB (for matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for visualization
    viz_image = image_rgb.copy()
    
    # Draw bounding boxes for detected symbols
    if "symbols" in results:
        for symbol in results["symbols"]:
            # Get the bounding box
            box = symbol["position"]
            box = [(int(p[0]), int(p[1])) for p in box]
            
            # Draw the box
            for i in range(len(box)):
                pt1 = box[i]
                pt2 = box[(i + 1) % len(box)]
                cv2.line(viz_image, pt1, pt2, (0, 255, 0), 2)
            
            # Draw the text
            text = symbol.get("math_symbol", symbol["text"])
            confidence = symbol["confidence"]
            cv2.putText(viz_image, f"{text} ({confidence:.2f})", 
                        (box[0][0], box[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Plot the original and visualized images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Detection Results")
    plt.imshow(viz_image)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to test handwriting recognition."""
    parser = argparse.ArgumentParser(description="Test handwriting recognition for mathematical notation")
    parser.add_argument("--image", "-i", type=str, help="Path to the image file")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the results")
    args = parser.parse_args()
    
    if not args.image:
        print("Please provide an image path using the --image argument")
        return
    
    print_separator("Handwriting Recognition Test")
    print(f"Processing image: {args.image}")
    
    # Process the image
    results = process_handwritten_image(args.image)
    
    # Print the results
    print_separator("Recognition Results")
    print(f"Success: {results['success']}")
    
    if results['success']:
        print(f"Symbol Count: {results['symbol_count']}")
        print(f"Confidence: {results['confidence']:.2f}")
        print("\nLaTeX:")
        print(results['latex'])
        print("\nDisplay LaTeX:")
        print(results['display_latex'])
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    # Visualize the results if requested
    if args.visualize and results['success']:
        visualize_results(args.image, results)
    
    print_separator("Test Completed")

if __name__ == "__main__":
    main()

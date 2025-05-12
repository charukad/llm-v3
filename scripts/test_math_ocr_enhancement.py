#!/usr/bin/env python3
"""
Test script for enhanced mathematical OCR capabilities implemented in Sprint 10.
"""

import os
import sys
import logging
import argparse
import cv2
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our OCR components
from multimodal.image_processing.preprocessor import ImagePreprocessor
from multimodal.ocr.advanced_symbol_detector import MathSymbolDetector
from multimodal.ocr.context_analyzer import MathContextAnalyzer
from multimodal.ocr.performance_optimizer import OCRPerformanceOptimizer
from multimodal.structure.layout_analyzer import MathLayoutAnalyzer
from multimodal.latex_generator.latex_generator import LaTeXGenerator
from multimodal.agent.ocr_agent import HandwritingRecognitionAgent

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def test_image_processing(image_path):
    """Test the complete image processing pipeline."""
    logging.info(f"Testing image processing with {image_path}")
    
    # Initialize the agent
    agent = HandwritingRecognitionAgent()
    
    # Process the image
    start_time = time.time()
    result = agent.process_image(image_path)
    elapsed = time.time() - start_time
    
    # Print results
    logging.info(f"Processing completed in {elapsed:.3f} seconds")
    logging.info(f"LaTeX result: {result['latex']}")
    logging.info(f"Confidence: {result['confidence']:.2f}")
    
    return result

def test_component(image_path, component_name):
    """Test a specific component."""
    logging.info(f"Testing component {component_name} with {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image from {image_path}")
        return
    
    # Initialize base preprocessor
    preprocessor = ImagePreprocessor()
    preprocessed = preprocessor.preprocess(image)
    
    # Test specific component
    if component_name == "advanced_detector":
        detector = MathSymbolDetector()
        start_time = time.time()
        symbols = detector.detect_symbols(preprocessed)
        elapsed = time.time() - start_time
        
        logging.info(f"Advanced symbol detection completed in {elapsed:.3f} seconds")
        logging.info(f"Detected {len(symbols)} symbols")
        for i, symbol in enumerate(symbols[:10]):  # Show only first 10
            logging.info(f"Symbol {i+1}: {symbol['text']} (conf: {symbol['confidence']:.2f})")
    
    elif component_name == "layout_analyzer":
        # First detect symbols
        detector = MathSymbolDetector()
        symbols = detector.detect_symbols(preprocessed)
        
        # Then analyze layout
        analyzer = MathLayoutAnalyzer()
        start_time = time.time()
        structure = analyzer.analyze_structure(symbols)
        elapsed = time.time() - start_time
        
        logging.info(f"Layout analysis completed in {elapsed:.3f} seconds")
        logging.info(f"Structure type: {structure.get('type', 'unknown')}")
        logging.info(f"Structure: {structure}")
    
    elif component_name == "context_analyzer":
        # First detect symbols
        detector = MathSymbolDetector()
        symbols = detector.detect_symbols(preprocessed)
        
        # Then apply context correction
        analyzer = MathContextAnalyzer()
        start_time = time.time()
        corrected = analyzer.analyze_and_correct(symbols)
        elapsed = time.time() - start_time
        
        logging.info(f"Context analysis completed in {elapsed:.3f} seconds")
        logging.info(f"Corrected {sum(1 for s in corrected if s.get('text') != s.get('original_text', s['text']))} symbols")
        
        # Show corrections
        for i, symbol in enumerate(corrected):
            original = symbol.get('original_text', symbol['text'])
            if original != symbol['text']:
                logging.info(f"Correction: '{original}' -> '{symbol['text']}'")
    
    elif component_name == "performance":
        optimizer = OCRPerformanceOptimizer()
        start_time = time.time()
        optimized = optimizer.optimize_image(image)
        elapsed = time.time() - start_time
        
        logging.info(f"Image optimization completed in {elapsed:.3f} seconds")
        logging.info(f"Original size: {image.shape[:2]}, Optimized size: {optimized.shape[:2]}")
    
    else:
        logging.error(f"Unknown component: {component_name}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test mathematical OCR enhancements")
    parser.add_argument("image_path", help="Path to the test image")
    parser.add_argument(
        "--component", 
        choices=["advanced_detector", "layout_analyzer", "context_analyzer", "performance"],
        help="Test a specific component instead of the full pipeline"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    if not os.path.exists(args.image_path):
        logging.error(f"Image not found: {args.image_path}")
        return 1
    
    if args.component:
        test_component(args.image_path, args.component)
    else:
        test_image_processing(args.image_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
Advanced symbol detector with enhanced capabilities for mathematical notation.

This module extends the basic symbol detector with more sophisticated
recognition techniques for complex mathematical symbols.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import numpy as np
import cv2
import json

from .symbol_detector import detect_symbols as basic_detect_symbols

logger = logging.getLogger(__name__)

# Load additional symbol maps if available
ADVANCED_SYMBOL_MAP_PATH = os.path.join(os.path.dirname(__file__), 'data', 'advanced_math_symbol_map.json')
ADVANCED_SYMBOL_MAP = {}

if os.path.exists(ADVANCED_SYMBOL_MAP_PATH):
    try:
        with open(ADVANCED_SYMBOL_MAP_PATH, 'r') as f:
            ADVANCED_SYMBOL_MAP = json.load(f)
    except Exception as e:
        logger.error(f"Error loading advanced symbol map: {str(e)}")
else:
    logger.warning(f"Advanced symbol map not found at {ADVANCED_SYMBOL_MAP_PATH}")

def detect_symbols(image_path: str) -> List[Dict[str, Any]]:
    """
    Detect mathematical symbols in an image with advanced techniques.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of detected symbols with positions and confidence scores
    """
    try:
        # Get base symbols from basic detector
        base_symbols = basic_detect_symbols(image_path)
        
        # Read the image for advanced processing
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return base_symbols
        
        # Enhance symbols with advanced processing
        enhanced_symbols = enhance_symbol_detection(image, base_symbols)
        
        # Detect special groupings (fractions, superscripts, etc.)
        enhanced_symbols = detect_special_groupings(image, enhanced_symbols)
        
        return enhanced_symbols
        
    except Exception as e:
        logger.error(f"Error in advanced symbol detection: {str(e)}")
        return []

def enhance_symbol_detection(image, base_symbols):
    """
    Enhance basic symbol detection with advanced techniques.
    
    Args:
        image: Image data
        base_symbols: Symbols detected by basic detector
        
    Returns:
        Enhanced list of symbols
    """
    # In a real implementation, this would use more sophisticated techniques
    # For this example, we'll just improve confidence based on context
    
    enhanced_symbols = []
    
    for i, symbol in enumerate(base_symbols):
        # Clone the symbol data
        enhanced_symbol = symbol.copy()
        
        # Improve confidence based on context (simplified)
        if i > 0 and i < len(base_symbols) - 1:
            # Symbols in the middle of a sequence are more likely correct
            enhanced_symbol["confidence"] = min(symbol["confidence"] + 0.1, 1.0)
        
        # Additional enhancements would be applied here
        
        enhanced_symbols.append(enhanced_symbol)
    
    return enhanced_symbols

def detect_special_groupings(image, symbols):
    """
    Detect special groupings like fractions, superscripts, etc.
    
    Args:
        image: Image data
        symbols: Basic detected symbols
        
    Returns:
        Symbols with special grouping information
    """
    # Process symbol positions to detect spatial relationships
    
    # Detect superscripts (symbols that are smaller and higher)
    for i in range(len(symbols) - 1):
        current = symbols[i]
        next_symbol = symbols[i + 1]
        
        # Check if next symbol is smaller and higher than current
        current_pos = current["position"]
        next_pos = next_symbol["position"]
        
        # Simple check for superscript
        if (next_pos[1] < current_pos[1] and              # Higher
            next_pos[2] < current_pos[2] * 0.8 and        # Smaller width
            next_pos[3] < current_pos[3] * 0.8 and        # Smaller height
            next_pos[0] >= current_pos[0]):               # To the right
            
            # Mark the next symbol as a superscript of the current
            next_symbol["relation"] = "superscript"
            next_symbol["related_to"] = i
    
    # Detect subscripts (symbols that are smaller and lower)
    for i in range(len(symbols) - 1):
        current = symbols[i]
        next_symbol = symbols[i + 1]
        
        # Check if next symbol is smaller and lower than current
        current_pos = current["position"]
        next_pos = next_symbol["position"]
        
        # Simple check for subscript
        if (next_pos[1] > current_pos[1] + current_pos[3] * 0.5 and  # Lower
            next_pos[2] < current_pos[2] * 0.8 and                  # Smaller width
            next_pos[3] < current_pos[3] * 0.8 and                  # Smaller height
            next_pos[0] >= current_pos[0]):                         # To the right
            
            # Mark the next symbol as a subscript of the current
            next_symbol["relation"] = "subscript"
            next_symbol["related_to"] = i
    
    # Detect horizontal lines that might be fraction bars
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50, 
        minLineLength=20, maxLineGap=10
    )
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is horizontal
            if abs(y2 - y1) < 5 and abs(x2 - x1) > 20:
                # Mark symbols above and below the line
                above_symbols = []
                below_symbols = []
                
                for i, symbol in enumerate(symbols):
                    pos = symbol["position"]
                    symbol_center_y = pos[1] + pos[3] / 2
                    
                    # Check if symbol is within the horizontal span of the line
                    if pos[0] < max(x1, x2) and pos[0] + pos[2] > min(x1, x2):
                        if symbol_center_y < min(y1, y2):
                            above_symbols.append(i)
                        elif symbol_center_y > max(y1, y2):
                            below_symbols.append(i)
                
                # If we have symbols both above and below, it's likely a fraction
                if above_symbols and below_symbols:
                    # Mark numerator and denominator symbols
                    for i in above_symbols:
                        symbols[i]["relation"] = "numerator"
                        symbols[i]["fraction_line"] = [x1, y1, x2, y2]
                        
                    for i in below_symbols:
                        symbols[i]["relation"] = "denominator"
                        symbols[i]["fraction_line"] = [x1, y1, x2, y2]
    
    return symbols

"""
Symbol detector for recognizing mathematical symbols.

This module detects and recognizes individual mathematical symbols
in preprocessed images.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import numpy as np
import cv2
import json

logger = logging.getLogger(__name__)

# Load symbol map if available
SYMBOL_MAP_PATH = os.path.join(os.path.dirname(__file__), 'data', 'math_symbol_map.json')
SYMBOL_MAP = {}

if os.path.exists(SYMBOL_MAP_PATH):
    try:
        with open(SYMBOL_MAP_PATH, 'r') as f:
            SYMBOL_MAP = json.load(f)
    except Exception as e:
        logger.error(f"Error loading symbol map: {str(e)}")
else:
    logger.warning(f"Symbol map not found at {SYMBOL_MAP_PATH}")

def detect_symbols(image_path: str) -> List[Dict[str, Any]]:
    """
    Detect mathematical symbols in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of detected symbols with positions and confidence scores
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate text from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right and top to bottom
        def sort_key(contour):
            x, y, w, h = cv2.boundingRect(contour)
            # Sort primarily by y, then by x (to read left-to-right, top-to-bottom)
            return (y // 20) * 10000 + x  # Group by rows (assuming line height of ~20 pixels)
        
        contours = sorted(contours, key=sort_key)
        
        # Process each contour to extract symbol
        symbols = []
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small contours (noise)
            if w < 5 or h < 5:
                continue
            
            # Extract the symbol
            symbol_image = binary[y:y+h, x:x+w]
            
            # Create a simplified feature vector for the symbol
            # In a real implementation, this would use a trained classifier
            # For this example, we'll use a simple approach based on aspect ratio and pixel density
            
            aspect_ratio = w / h if h > 0 else 0
            pixel_density = np.sum(symbol_image) / (w * h * 255) if w * h > 0 else 0
            
            # Simple classification based on these features
            symbol_text, confidence = classify_symbol(symbol_image, aspect_ratio, pixel_density)
            
            # Add to symbols list
            symbols.append({
                "text": symbol_text,
                "position": [int(x), int(y), int(w), int(h)],
                "confidence": float(confidence),
                "features": {
                    "aspect_ratio": float(aspect_ratio),
                    "pixel_density": float(pixel_density)
                }
            })
        
        return symbols
        
    except Exception as e:
        logger.error(f"Error detecting symbols: {str(e)}")
        return []

def classify_symbol(symbol_image, aspect_ratio, pixel_density):
    """
    Classify a symbol based on simple features.
    
    Args:
        symbol_image: Image of the symbol
        aspect_ratio: Width-to-height ratio
        pixel_density: Ratio of foreground pixels to total pixels
        
    Returns:
        Tuple of (symbol_text, confidence)
    """
    # In a real implementation, this would use a trained classifier
    # For this example, we'll use a simple rule-based approach
    
    # Feature-based classification (very simplified)
    if aspect_ratio < 0.5:
        # Tall and narrow: likely characters like 1, l, i, etc.
        if pixel_density < 0.3:
            return "1", 0.8
        else:
            return "i", 0.7
    elif aspect_ratio > 1.5:
        # Wide and short: likely characters like -, =, etc.
        if pixel_density < 0.3:
            return "-", 0.8
        else:
            return "=", 0.7
    elif 0.9 < aspect_ratio < 1.1:
        # Nearly square: likely characters like +, x, etc.
        if pixel_density < 0.2:
            return "+", 0.7
        elif pixel_density < 0.4:
            return "x", 0.7
        else:
            return "0", 0.7
    else:
        # Default to common symbols
        if pixel_density < 0.3:
            return "x", 0.6
        else:
            return "a", 0.5

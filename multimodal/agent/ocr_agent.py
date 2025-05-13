"""
OCR Agent for processing handwritten mathematical notation.

This agent processes images containing handwritten mathematical notation,
recognizes symbols, and converts them to LaTeX format.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import time

from ..ocr.symbol_detector import detect_symbols
from ..structure.layout_analyzer import analyze_layout
from ..latex_generator.latex_generator import generate_latex

logger = logging.getLogger(__name__)

class OCRAgent:
    """Agent for handwritten mathematics recognition."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCR agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.min_confidence_threshold = self.config.get("min_confidence", 0.6)
        logger.info("Initialized OCR agent")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process handwritten mathematical input.
        
        Args:
            input_data: Dictionary containing processed image data
            
        Returns:
            Processing result with recognized LaTeX
        """
        start_time = time.time()
        
        try:
            # Extract necessary data
            image_data = input_data.get("preprocessed_image", input_data)
            image_path = image_data.get("image_path", None)
            
            # If we don't have preprocessed symbols, detect them
            symbols = image_data.get("symbols", None)
            if not symbols:
                if not image_path or not os.path.exists(image_path):
                    return {
                        "success": False,
                        "error": "No image path provided or file does not exist"
                    }
                symbols = detect_symbols(image_path)
            
            # Filter low-confidence symbols
            filtered_symbols = [
                symbol for symbol in symbols 
                if symbol.get("confidence", 0) >= self.min_confidence_threshold
            ]
            
            if not filtered_symbols:
                return {
                    "success": False,
                    "error": "No symbols detected with sufficient confidence"
                }
            
            # If we don't have a layout structure, analyze it
            structure = image_data.get("structure", None)
            if not structure:
                structure = analyze_layout(filtered_symbols)
            
            # Generate LaTeX from the structure
            latex = image_data.get("recognized_latex", None)
            if not latex:
                latex = generate_latex(structure)
            
            # Calculate overall confidence score
            if "confidence" not in image_data:
                confidence_scores = [s.get("confidence", 0) for s in filtered_symbols]
                overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            else:
                overall_confidence = image_data.get("confidence")
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "recognized_latex": latex,
                "structure": structure,
                "symbols": filtered_symbols,
                "confidence": overall_confidence,
                "processing_time_ms": round(processing_time * 1000, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in OCR agent: {str(e)}")
            return {
                "success": False,
                "error": f"OCR processing error: {str(e)}"
            }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from the message bus.
        
        Args:
            message: Message from the message bus
            
        Returns:
            Processing result
        """
        # Extract message body
        body = message.get("body", {})
        
        # Extract input data from message body
        input_data = body.get("processed_input", body)
        
        # Process the input
        result = self.process(input_data)
        
        # Prepare response
        response = {
            "success": result.get("success", False),
            "input_type": "image",
            "agent_type": "ocr"
        }
        
        # Include result data
        if result.get("success", False):
            response.update({
                "recognized_latex": result.get("recognized_latex", ""),
                "confidence": result.get("confidence", 0),
                "processing_time_ms": result.get("processing_time_ms", 0)
            })
        else:
            response["error"] = result.get("error", "Unknown error")
        
        return response

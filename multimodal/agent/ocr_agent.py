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
from ..structure.layout_analyzer import MathLayoutAnalyzer
from ..latex_generator.latex_generator import generate_latex
from ..image_processing.preprocessor import preprocess_image

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
        self.layout_analyzer = MathLayoutAnalyzer()
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
                structure = self.layout_analyzer.analyze(filtered_symbols)
            
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

class HandwritingRecognitionAgent(OCRAgent):
    """
    Agent specialized for handwritten mathematical expressions.
    
    This agent extends the basic OCR agent with capabilities 
    specifically tuned for handwritten mathematical notation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the handwriting recognition agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.min_confidence_threshold = self.config.get("min_confidence", 0.5)  # Lower threshold for handwriting
        self.preprocessing_options = self.config.get("preprocessing_options", {
            "enhance_contrast": True,
            "remove_noise": True,
            "correct_skew": True
        })
        logger.info("Initialized handwriting recognition agent")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process handwritten mathematical input with specialized handling.
        
        Args:
            input_data: Dictionary containing processed image data
            
        Returns:
            Processing result with recognized LaTeX
        """
        # Apply enhanced preprocessing specific to handwriting
        enhanced_input = self._enhance_for_handwriting(input_data)
        
        # Use the base OCR process method with enhanced input
        result = super().process(enhanced_input)
        
        # Add handwriting-specific metadata if successful
        if result.get("success", False):
            result["agent_type"] = "handwriting_recognition"
            result["preprocessing_applied"] = self.preprocessing_options
        
        return result
    
    def _enhance_for_handwriting(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply handwriting-specific enhancements to the input data.
        
        Args:
            input_data: Original input data
            
        Returns:
            Enhanced input data optimized for handwriting recognition
        """
        # Clone the input data to avoid modifying the original
        enhanced_data = input_data.copy()
        
        # If we have an image path but no preprocessed image, apply preprocessing
        image_path = enhanced_data.get("image_path")
        if image_path and os.path.exists(image_path):
            try:
                # Apply preprocessing with handwriting-specific options
                preprocessed = preprocess_image(
                    image_path,
                    enhance_contrast=self.preprocessing_options.get("enhance_contrast", True),
                    remove_noise=self.preprocessing_options.get("remove_noise", True),
                    correct_skew=self.preprocessing_options.get("correct_skew", True)
                )
                
                enhanced_data["preprocessed_image"] = preprocessed
            except Exception as e:
                logger.warning(f"Error in handwriting enhancement: {str(e)}")
        
        return enhanced_data

def process_handwritten_image(image_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a handwritten image and recognize mathematical content.
    
    This is a convenience function for processing a single image without
    creating an agent instance.
    
    Args:
        image_path: Path to the image file
        config: Optional configuration dictionary
        
    Returns:
        Processing result with recognized LaTeX
    """
    agent = HandwritingRecognitionAgent(config)
    return agent.process({"image_path": image_path})

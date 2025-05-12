"""
Advanced OCR Agent for processing complex mathematical content.

This agent extends the basic OCR agent with capabilities for processing
diagrams, coordinate systems, and multiple input formats.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import time

from .ocr_agent import OCRAgent
from ..ocr.advanced_symbol_detector import detect_symbols as advanced_detect_symbols
from ..image_processing.diagram_detector import detect_diagrams
from ..image_processing.coordinate_detector import detect_coordinate_system
from ..image_processing.format_handler import detect_format, convert_format

logger = logging.getLogger(__name__)

class AdvancedOCRAgent:
    """Agent for advanced mathematical image processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced OCR agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize base OCR agent for standard processing
        self.ocr_agent = OCRAgent(config)
        
        # Additional configurations
        self.process_diagrams = self.config.get("process_diagrams", True)
        self.process_coordinates = self.config.get("process_coordinates", True)
        
        logger.info("Initialized advanced OCR agent")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complex mathematical content in images.
        
        Args:
            input_data: Dictionary containing processed image data
            
        Returns:
            Processing result with recognized content
        """
        start_time = time.time()
        
        try:
            # Extract necessary data
            image_data = input_data.get("preprocessed_image", input_data)
            image_path = image_data.get("image_path", None)
            
            if not image_path or not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": "No image path provided or file does not exist"
                }
            
            # Detect and handle format
            format_info = detect_format(image_path)
            
            # If not standard image format, convert
            if format_info.get("format") != "standard":
                conversion_result = convert_format(image_path, format_info)
                if conversion_result.get("success", False):
                    image_path = conversion_result.get("converted_path")
                else:
                    logger.warning(f"Format conversion failed: {conversion_result.get('error')}")
            
            # Initialize results dictionary
            results = {
                "success": True,
                "image_path": image_path,
                "format": format_info.get("format", "unknown"),
                "content_types": []
            }
            
            # Check for diagrams if enabled
            if self.process_diagrams:
                diagram_result = detect_diagrams(image_path)
                if diagram_result.get("has_diagrams", False):
                    results["diagrams"] = diagram_result.get("diagrams", [])
                    results["content_types"].append("diagram")
            
            # Check for coordinate systems if enabled
            if self.process_coordinates:
                coordinate_result = detect_coordinate_system(image_path)
                if coordinate_result.get("has_coordinates", False):
                    results["coordinates"] = coordinate_result
                    results["content_types"].append("coordinates")
            
            # Use advanced symbol detection
            symbols = advanced_detect_symbols(image_path)
            results["symbols"] = symbols
            
            # If we have mathematical expressions, add to content types
            if symbols:
                results["content_types"].append("expressions")
            
            # If we have expression content, process with base OCR agent
            if "expressions" in results["content_types"]:
                # Prepare input for OCR agent
                ocr_input = {
                    "image_path": image_path,
                    "symbols": symbols
                }
                
                ocr_result = self.ocr_agent.process(ocr_input)
                
                # Merge OCR results
                if ocr_result.get("success", False):
                    results["recognized_latex"] = ocr_result.get("recognized_latex", "")
                    results["structure"] = ocr_result.get("structure", {})
                    results["confidence"] = ocr_result.get("confidence", 0)
            
            # Determine overall success based on found content
            if not results["content_types"]:
                results["success"] = False
                results["error"] = "No recognizable content found in image"
            
            processing_time = time.time() - start_time
            results["processing_time_ms"] = round(processing_time * 1000, 2)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced OCR agent: {str(e)}")
            return {
                "success": False,
                "error": f"Advanced OCR processing error: {str(e)}"
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
            "agent_type": "advanced_ocr"
        }
        
        # Include result data
        if result.get("success", False):
            response.update({
                "content_types": result.get("content_types", []),
                "processing_time_ms": result.get("processing_time_ms", 0)
            })
            
            # Include content-specific results
            if "expressions" in result.get("content_types", []):
                response["recognized_latex"] = result.get("recognized_latex", "")
                response["confidence"] = result.get("confidence", 0)
            
            if "diagram" in result.get("content_types", []):
                response["diagrams"] = result.get("diagrams", [])
            
            if "coordinates" in result.get("content_types", []):
                response["coordinates"] = result.get("coordinates", {})
        else:
            response["error"] = result.get("error", "Unknown error")
        
        return response

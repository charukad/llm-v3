"""
Core input processor for unified multimodal processing.

This module provides the main implementation of the unified input pipeline,
responsible for detecting input types, routing to appropriate processors,
and coordinating the results.
"""
import os
import mimetypes
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from ..image_processing.preprocessor import preprocess_image
from ..image_processing.format_handler import detect_format, convert_format
from ..ocr.advanced_symbol_detector import detect_symbols
from ..structure.layout_analyzer import analyze_layout
from ..latex_generator.latex_generator import generate_latex

logger = logging.getLogger(__name__)

class InputProcessor:
    """Unified input processor for multimodal content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the input processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.supported_image_formats = [
            'image/png', 'image/jpeg', 'image/jpg', 'image/gif',
            'application/pdf', 'image/tiff'
        ]
        self.supported_text_formats = [
            'text/plain', 'text/markdown', 'text/x-latex'
        ]
        logger.info("Initialized unified input processor")
    
    def process_input(self, input_data: Union[str, bytes, Dict[str, Any]], 
                     input_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process input data of any supported type.
        
        Args:
            input_data: The input data, which could be text, binary data, or a dictionary
            input_type: Optional explicit input type
            
        Returns:
            Dictionary containing processed results
        """
        # Determine input type if not provided
        if input_type is None:
            input_type = self._detect_input_type(input_data)
        
        logger.info(f"Processing input of type: {input_type}")
        
        # Route to appropriate processor
        if input_type.startswith('image/') or input_type == 'application/pdf':
            return self._process_image_input(input_data, input_type)
        elif input_type.startswith('text/'):
            return self._process_text_input(input_data, input_type)
        elif input_type == 'multipart/mixed':
            return self._process_multipart_input(input_data)
        else:
            error_msg = f"Unsupported input type: {input_type}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def _detect_input_type(self, input_data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Detect the type of input data.
        
        Args:
            input_data: The input data to analyze
            
        Returns:
            String representing the detected input type
        """
        if isinstance(input_data, str):
            # Check if it's a path to a file
            if os.path.isfile(input_data):
                mime_type, _ = mimetypes.guess_type(input_data)
                return mime_type or 'text/plain'
            else:
                # Assume it's plain text
                return 'text/plain'
        elif isinstance(input_data, bytes):
            # Try to detect file type from bytes
            # This is a simplified approach - in a real implementation,
            # you'd use more sophisticated file type detection
            if input_data.startswith(b'%PDF'):
                return 'application/pdf'
            elif input_data.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'image/png'
            elif input_data.startswith(b'\xff\xd8'):
                return 'image/jpeg'
            else:
                return 'application/octet-stream'
        elif isinstance(input_data, dict):
            # If it's a dictionary, check for type information
            if 'type' in input_data:
                return input_data['type']
            elif len(input_data) > 1:
                return 'multipart/mixed'
            else:
                return 'application/json'
        else:
            return 'unknown/unknown'
    
    def _process_image_input(self, image_data: Union[str, bytes], 
                            image_type: str) -> Dict[str, Any]:
        """
        Process image input data.
        
        Args:
            image_data: Path to image file or image binary data
            image_type: The mime type of the image
            
        Returns:
            Dictionary containing processed results
        """
        try:
            # Load and preprocess the image
            if isinstance(image_data, str):
                # It's a file path
                image_path = image_data
            else:
                # Save binary data to a temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_extension(image_type)) as temp:
                    temp.write(image_data)
                    image_path = temp.name
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image_path)
            
            # Detect symbols in the image
            symbols = detect_symbols(preprocessed_image)
            
            # Analyze the layout structure
            structure = analyze_layout(symbols)
            
            # Generate LaTeX from the structure
            latex = generate_latex(structure)
            
            # Cleanup temporary file if created
            if isinstance(image_data, bytes) and os.path.exists(image_path):
                os.remove(image_path)
            
            return {
                "success": True,
                "input_type": "image",
                "image_type": image_type,
                "recognized_latex": latex,
                "structure": structure,
                "symbols": symbols,
                "confidence": self._calculate_confidence(symbols)
            }
        
        except Exception as e:
            logger.error(f"Error processing image input: {str(e)}")
            return {
                "success": False,
                "input_type": "image",
                "error": str(e)
            }
    
    def _process_text_input(self, text_data: Union[str, bytes, Dict[str, Any]], 
                           text_type: str) -> Dict[str, Any]:
        """
        Process text input data.
        
        Args:
            text_data: The text data to process
            text_type: The mime type of the text
            
        Returns:
            Dictionary containing processed results
        """
        try:
            # Ensure text is string
            if isinstance(text_data, bytes):
                text = text_data.decode('utf-8')
            elif isinstance(text_data, dict):
                text = text_data.get('content', '')
            else:
                text = text_data
            
            # Process based on text type
            if text_type == 'text/x-latex':
                # It's already LaTeX
                return {
                    "success": True,
                    "input_type": "text",
                    "text_type": "latex",
                    "latex": text,
                    "confidence": 1.0
                }
            elif text_type == 'text/plain':
                # Check if it contains mathematical notation
                # (In a real implementation, this would be more sophisticated)
                if any(symbol in text for symbol in ['\\', '^', '_', '\\frac', '\\sum', '\\int']):
                    return {
                        "success": True,
                        "input_type": "text",
                        "text_type": "latex",
                        "latex": text,
                        "confidence": 0.9
                    }
                else:
                    return {
                        "success": True,
                        "input_type": "text",
                        "text_type": "plain",
                        "text": text,
                        "confidence": 1.0
                    }
            else:
                return {
                    "success": True,
                    "input_type": "text",
                    "text_type": text_type.split('/')[-1],
                    "text": text,
                    "confidence": 1.0
                }
                
        except Exception as e:
            logger.error(f"Error processing text input: {str(e)}")
            return {
                "success": False,
                "input_type": "text",
                "error": str(e)
            }
    
    def _process_multipart_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multipart input containing multiple modalities.
        
        Args:
            input_data: Dictionary containing multiple inputs
            
        Returns:
            Dictionary containing processed results from all inputs
        """
        results = {}
        
        try:
            for key, value in input_data.items():
                if key == 'type':
                    continue
                
                # Detect type if not explicitly provided
                part_type = value.get('type') if isinstance(value, dict) else None
                if part_type is None:
                    part_data = value.get('content', value) if isinstance(value, dict) else value
                    part_type = self._detect_input_type(part_data)
                
                # Process each part
                part_result = self.process_input(
                    value.get('content', value) if isinstance(value, dict) else value,
                    part_type
                )
                
                results[key] = part_result
            
            return {
                "success": True,
                "input_type": "multipart",
                "parts": results
            }
            
        except Exception as e:
            logger.error(f"Error processing multipart input: {str(e)}")
            return {
                "success": False,
                "input_type": "multipart",
                "error": str(e),
                "partial_results": results
            }
    
    def _calculate_confidence(self, symbols: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence score for recognized symbols.
        
        Args:
            symbols: List of recognized symbols with confidence scores
            
        Returns:
            Overall confidence score (0.0-1.0)
        """
        if not symbols:
            return 0.0
        
        # Average confidence across all symbols
        confidences = [symbol.get('confidence', 0.0) for symbol in symbols]
        return sum(confidences) / len(confidences)
    
    def _get_extension(self, mime_type: str) -> str:
        """
        Get file extension for a mime type.
        
        Args:
            mime_type: The mime type
            
        Returns:
            File extension including the dot
        """
        extensions = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/gif': '.gif',
            'application/pdf': '.pdf',
            'image/tiff': '.tiff'
        }
        
        return extensions.get(mime_type, '.tmp')

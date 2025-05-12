"""
Content router for directing input to appropriate processing agents.

This module determines which agents should process each input type
and manages the routing of messages between them.
"""
import logging
from typing import Dict, Any, List, Optional, Union

from ..agent.ocr_agent import OCRAgent
from ..agent.advanced_ocr_agent import AdvancedOCRAgent

logger = logging.getLogger(__name__)

class ContentRouter:
    """Routes content to appropriate processing agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the content router.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.ocr_agent = OCRAgent()
        self.advanced_ocr_agent = AdvancedOCRAgent()
        logger.info("Initialized content router")
    
    def route_content(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route processed input to appropriate agents for further processing.
        
        Args:
            processed_input: Dictionary containing processed input data
            
        Returns:
            Dictionary containing agent processing results
        """
        input_type = processed_input.get('input_type')
        
        if not input_type:
            error_msg = "Input type not specified in processed input"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
        
        logger.info(f"Routing content of type: {input_type}")
        
        if input_type == 'image':
            return self._route_image_content(processed_input)
        elif input_type == 'text':
            return self._route_text_content(processed_input)
        elif input_type == 'multipart':
            return self._route_multipart_content(processed_input)
        else:
            error_msg = f"Unsupported content type for routing: {input_type}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def _route_image_content(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route image content to appropriate agents.
        
        Args:
            image_data: Dictionary containing processed image data
            
        Returns:
            Dictionary containing agent processing results
        """
        # Check if we need basic OCR or advanced OCR
        image_type = image_data.get('image_type', '')
        
        if image_type == 'application/pdf' or 'diagram' in image_data.get('structure', {}):
            # Use advanced OCR for PDFs and diagrams
            agent_result = self.advanced_ocr_agent.process(image_data)
        else:
            # Use basic OCR for simple handwritten math
            agent_result = self.ocr_agent.process(image_data)
        
        return {
            "success": agent_result.get('success', False),
            "source_type": "image",
            "agent_type": "ocr" if image_type != 'application/pdf' else "advanced_ocr",
            "result": agent_result
        }
    
    def _route_text_content(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route text content to appropriate agents.
        
        Args:
            text_data: Dictionary containing processed text data
            
        Returns:
            Dictionary containing routing results
        """
        # For text content, we typically route to the Core LLM Agent or Math Agent
        # This would be handled by the orchestration layer, so we just return
        # the processed data for now
        return {
            "success": True,
            "source_type": "text",
            "agent_type": "core_llm",  # This would be determined by orchestration
            "result": text_data
        }
    
    def _route_multipart_content(self, multipart_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route multipart content to appropriate agents.
        
        Args:
            multipart_data: Dictionary containing processed multipart data
            
        Returns:
            Dictionary containing routing results for all parts
        """
        parts = multipart_data.get('parts', {})
        results = {}
        
        for key, part_data in parts.items():
            # Route each part based on its type
            part_type = part_data.get('input_type')
            
            if part_type == 'image':
                results[key] = self._route_image_content(part_data)
            elif part_type == 'text':
                results[key] = self._route_text_content(part_data)
            else:
                results[key] = {
                    "success": False,
                    "error": f"Unknown part type: {part_type}"
                }
        
        return {
            "success": True,
            "source_type": "multipart",
            "parts": results
        }

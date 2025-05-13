"""
Content router for directing input to appropriate processing agents.

This module determines which agents should process each input type
and manages the routing of messages between them.
"""
import logging
from typing import Dict, Any, List, Optional, Union

from ..agent.ocr_agent import OCRAgent
from ..agent.advanced_ocr_agent import AdvancedOCRAgent
from ..agent.llm_router_agent import LLMRouterAgent

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
        
        # Initialize LLM Router agent for intelligent routing
        use_llm_router = self.config.get("use_llm_router", True)
        if use_llm_router:
            self.llm_router_agent = LLMRouterAgent(self.config.get("llm_router_config"))
            logger.info("Initialized content router with LLM-powered routing")
        else:
            self.llm_router_agent = None
            logger.info("Initialized content router with rule-based routing")
    
    def route_content(self, processed_input: Dict[str, Any], 
                     context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route processed input to appropriate agents for further processing.
        
        Args:
            processed_input: Dictionary containing processed input data
            context_data: Optional context data including conversation history
            
        Returns:
            Dictionary containing agent processing results
        """
        input_type = processed_input.get('input_type')
        
        if not input_type:
            error_msg = "Input type not specified in processed input"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
        
        logger.info(f"Routing content of type: {input_type}")
        
        # Use LLM routing if available
        if self.llm_router_agent:
            return self._route_with_llm(processed_input, context_data)
        
        # Fall back to rule-based routing if LLM router is not available
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
    
    def _route_with_llm(self, processed_input: Dict[str, Any], 
                       context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use LLM to intelligently route content.
        
        Args:
            processed_input: Dictionary containing processed input data
            context_data: Optional context data
            
        Returns:
            Dictionary containing routing results
        """
        # Get routing decision from LLM router agent
        routing_result = self.llm_router_agent.route_request(processed_input, context_data)
        
        if not routing_result.get("success", False):
            logger.warning(f"LLM routing failed: {routing_result.get('error', 'Unknown error')}")
            # Fall back to rule-based routing
            input_type = processed_input.get('input_type')
            if input_type == 'image':
                return self._route_image_content(processed_input)
            elif input_type == 'text':
                return self._route_text_content(processed_input)
            elif input_type == 'multipart':
                return self._route_multipart_content(processed_input)
            else:
                return {"error": "Routing failed", "success": False}
        
        # Extract routing decision
        routing_decision = routing_result.get("routing_decision", {})
        primary_agent = routing_decision.get("primary_agent", "core_llm")
        capabilities = routing_decision.get("capabilities_needed", [])
        confidence = routing_decision.get("confidence", 0.7)
        reasoning = routing_decision.get("reasoning", "")
        
        logger.info(f"LLM routing decision: {primary_agent} (confidence: {confidence})")
        logger.debug(f"Routing reasoning: {reasoning}")
        
        # Perform specific agent processing based on the routing decision
        if primary_agent == "ocr" and processed_input.get('input_type') == 'image':
            # For OCR, use existing OCR agents
            return self._route_image_content(processed_input)
        elif primary_agent == "math_computation":
            # For math computation, prepare for math agent
            return {
                "success": True,
                "source_type": processed_input.get('input_type'),
                "agent_type": "math_computation",
                "capabilities": capabilities,
                "confidence": confidence,
                "reasoning": reasoning,
                "result": processed_input
            }
        elif primary_agent == "visualization":
            # For visualization requests
            return {
                "success": True,
                "source_type": processed_input.get('input_type'),
                "agent_type": "visualization",
                "capabilities": capabilities,
                "confidence": confidence,
                "reasoning": reasoning,
                "result": processed_input
            }
        elif primary_agent == "search":
            # For search requests
            return {
                "success": True,
                "source_type": processed_input.get('input_type'), 
                "agent_type": "search",
                "capabilities": capabilities,
                "confidence": confidence,
                "reasoning": reasoning,
                "result": processed_input
            }
        else:
            # Default to core_llm for everything else
            return {
                "success": True,
                "source_type": processed_input.get('input_type'),
                "agent_type": "core_llm",
                "capabilities": capabilities,
                "confidence": confidence,
                "reasoning": reasoning,
                "result": processed_input
            }
    
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

"""
Core LLM Agent implementation using Mistral 7B.

This module provides the main implementation of the Core LLM Agent,
which is responsible for natural language understanding, reasoning,
and response generation.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import time

from ..mistral.inference import InferenceEngine
from ..prompting.system_prompts import MATH_SYSTEM_PROMPT
from ..prompting.chain_of_thought import generate_cot_prompt

logger = logging.getLogger(__name__)

class CoreLLMAgent:
    """Core LLM Agent using Mistral 7B for natural language understanding and generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Core LLM Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Get settings from environment variables or config
        model_path = self.config.get("model_path", "mistralai/Mistral-7B-v0.1")
        lmstudio_url = self.config.get("lmstudio_url", os.environ.get('LMSTUDIO_URL', 'http://127.0.0.1:1234'))
        lmstudio_model = self.config.get("lmstudio_model", os.environ.get('LMSTUDIO_MODEL', 'mistral-7b-instruct-v0.3'))
        use_lmstudio_str = os.environ.get('USE_LMSTUDIO', '1')
        use_lmstudio = self.config.get("use_lmstudio", use_lmstudio_str == '1')
        
        logger.info(f"LLM Agent config: LMStudio enabled: {use_lmstudio}, URL: {lmstudio_url}, Model: {lmstudio_model}")
        
        # Initialize inference engine
        self.inference = InferenceEngine(
            model_path=model_path,
            use_lmstudio=use_lmstudio,
            lmstudio_url=lmstudio_url,
            lmstudio_model=lmstudio_model
        )
        
        logger.info(f"Initialized Core LLM Agent with model: {lmstudio_model if use_lmstudio else model_path}")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None,
                        use_cot: bool = True) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt to override default
            use_cot: Whether to use chain-of-thought prompting
            
        Returns:
            Dictionary containing the generated response
        """
        start_time = time.time()
        
        try:
            # Use default math system prompt if not provided
            if system_prompt is None:
                system_prompt = MATH_SYSTEM_PROMPT
            
            # Apply chain of thought if requested
            if use_cot:
                full_prompt = generate_cot_prompt(prompt, system_prompt)
            else:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate response
            response = self.inference.generate(full_prompt)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "response": response,
                "processing_time_ms": round(processing_time * 1000, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in LLM response generation: {str(e)}")
            return {
                "success": False,
                "error": f"Generation error: {str(e)}"
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
        
        # Extract prompt from message body
        prompt = body.get("prompt", "")
        system_prompt = body.get("system_prompt")
        use_cot = body.get("use_cot", True)
        
        if not prompt:
            return {
                "success": False,
                "error": "No prompt provided in message"
            }
        
        # Generate response
        result = self.generate_response(prompt, system_prompt, use_cot)
        
        # Add message metadata to result
        result["message_id"] = message.get("header", {}).get("message_id")
        result["message_type"] = message.get("header", {}).get("message_type")
        
        return result
    
    def classify_mathematical_domain(self, text: str) -> Dict[str, Any]:
        """
        Classify the mathematical domain of a text.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary containing the classification result
        """
        prompt = f"""Please classify the following mathematical query into one of these domains:
- algebra
- calculus
- geometry
- statistics
- probability
- linear_algebra
- number_theory
- discrete_mathematics
- other

Query: {text}

Respond with only the domain name and a confidence score between 0 and 1, in this format:
<domain>|<confidence>
"""
        
        # Generate response without chain of thought
        result = self.generate_response(prompt, use_cot=False)
        
        if not result.get("success", False):
            return {
                "success": False,
                "error": result.get("error", "Classification failed")
            }
        
        # Parse the response
        response = result.get("response", "")
        try:
            domain, confidence = response.strip().split("|")
            domain = domain.strip().lower()
            confidence = float(confidence)
            
            return {
                "success": True,
                "domain": domain,
                "confidence": confidence
            }
        except:
            # Fallback parsing
            for domain in ["algebra", "calculus", "geometry", "statistics", 
                          "probability", "linear_algebra", "number_theory", 
                          "discrete_mathematics", "other"]:
                if domain in response.lower():
                    return {
                        "success": True,
                        "domain": domain,
                        "confidence": 0.7,  # Default confidence if not parsed correctly
                        "note": "Confidence estimated due to parsing error"
                    }
            
            return {
                "success": False,
                "error": "Could not parse classification response",
                "response": response
            }
    
    def extract_mathematical_expressions(self, text: str) -> Dict[str, Any]:
        """
        Extract mathematical expressions from text.
        
        Args:
            text: Text to extract expressions from
            
        Returns:
            Dictionary containing the extracted expressions
        """
        prompt = f"""Please extract all mathematical expressions from the following text. 
For each expression, provide:
1. The raw expression as it appears in the text
2. A LaTeX representation of the expression

Text: {text}

Format your response as follows:
Expression 1:
Raw: <raw expression>
LaTeX: <latex representation>

Expression 2:
Raw: <raw expression>
LaTeX: <latex representation>

And so on. If no mathematical expressions are found, respond with "No expressions found."
"""
        
        # Generate response without chain of thought
        result = self.generate_response(prompt, use_cot=False)
        
        if not result.get("success", False):
            return {
                "success": False,
                "error": result.get("error", "Extraction failed")
            }
        
        # Parse the response
        response = result.get("response", "")
        
        if "No expressions found" in response:
            return {
                "success": True,
                "expressions": [],
                "expression_count": 0
            }
        
        # Parse expressions
        expressions = []
        expression_blocks = response.split("Expression")[1:]  # Skip the first split which is empty
        
        for block in expression_blocks:
            lines = block.strip().split("\n")
            expression = {}
            
            for line in lines:
                if line.startswith("Raw:"):
                    expression["raw"] = line[4:].strip()
                elif line.startswith("LaTeX:"):
                    expression["latex"] = line[6:].strip()
            
            if "raw" in expression and "latex" in expression:
                expressions.append(expression)
        
        return {
            "success": True,
            "expressions": expressions,
            "expression_count": len(expressions)
        }

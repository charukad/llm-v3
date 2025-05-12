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

from ..mistral.inference import MistralInference
from ..prompting.system_prompts import BASE_MATH_SYSTEM_PROMPT
from ..prompting.chain_of_thought import format_cot_prompt

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
        
        # Get LMStudio server URL from environment or config
        lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
        logger.info(f"Using LMStudio server at: {lmstudio_url}")
        
        # Initialize the inference engine with LMStudio server
        self.inference = MistralInference(
            api_url=lmstudio_url,
            n_ctx=2048  # Context window size
        )
        
        logger.info(f"Initialized Core LLM Agent with LMStudio server")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stop_sequences: Optional[List[str]] = None,
        use_cot: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: The prompt to respond to
            system_prompt: Optional system prompt to prepend
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional sequences to stop generation
            use_cot: Whether to use chain of thought (not implemented)
            
        Returns:
            Dictionary containing the response and success status
        """
        try:
            # Prepare full prompt
            if system_prompt is None:
                system_prompt = BASE_MATH_SYSTEM_PROMPT
            
            # Format full prompt for Mistral Instruct format
            full_prompt = f"<s>[INST] {system_prompt}\n\nQuestion: {prompt} [/INST]"
            
            # Generate response
            response = self.inference.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
            
            return {
                "success": True,
                "response": response.strip()
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_math_explanation(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        computation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a mathematical explanation for a query.
        
        Args:
            query: The mathematical query to explain
            query_analysis: Analysis of the query
            computation_result: Optional computation result
            
        Returns:
            Dictionary containing the explanation
        """
        # Prepare the prompt
        prompt_parts = [
            "Please provide a clear mathematical explanation for the following query.",
            f"\nQuery: {query}"
        ]
        
        if query_analysis:
            domain = query_analysis.get("domain", "")
            if domain:
                prompt_parts.append(f"\nMathematical Domain: {domain}")
            
            operations = query_analysis.get("operations", [])
            if operations:
                prompt_parts.append("\nRequired Operations:")
                for op in operations:
                    prompt_parts.append(f"- {op}")
        
        if computation_result:
            result = computation_result.get("result", "")
            steps = computation_result.get("steps", [])
            
            if result:
                prompt_parts.append(f"\nFinal Result: {result}")
            
            if steps:
                prompt_parts.append("\nComputation Steps:")
                for i, step in enumerate(steps, 1):
                    prompt_parts.append(f"{i}. {step}")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate the explanation
        explanation = self.generate_response(
            prompt=prompt,
            system_prompt="You are a mathematical expert providing clear and detailed explanations.",
            temperature=0.3,  # Slightly higher temperature for more natural explanations
            max_tokens=1024
        )
        
        return {
            "success": True,
            "explanation": explanation,
            "query": query
            }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from the message bus.
        
        Args:
            message: Message from the message bus
            
        Returns:
            Processing result
        """
        # Extract message body and type
        body = message.get("body", {})
        message_type = message.get("header", {}).get("message_type")
        
        if message_type == "generate_math_explanation":
            # Handle math explanation request
            query = body.get("query", "")
            query_analysis = body.get("analysis", {})
            computation_result = body.get("computation_result")
            
            if not query:
                return {
                    "success": False,
                    "error": "No query provided in message"
                }
            
            return self.generate_math_explanation(query, query_analysis, computation_result)
        
        # Handle other message types
        prompt = body.get("prompt", "")
        system_prompt = body.get("system_prompt")
        
        if not prompt:
            return {
                "success": False,
                "error": "No prompt provided in message"
            }
        
        # Generate response
        response = self.generate_response(prompt, system_prompt)
        
        return {
            "success": True,
            "response": response,
            "message_id": message.get("header", {}).get("message_id"),
            "message_type": message_type
        }
    
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

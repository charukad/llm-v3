"""
Integration between Core LLM Agent and multimodal components.

This module handles the integration between the Core LLM Agent and the
multimodal processing pipeline, allowing the LLM to process and reason
with inputs from different modalities.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import json

from .llm_agent import CoreLLMAgent

logger = logging.getLogger(__name__)

class MultimodalLLMIntegration:
    """Integration between Core LLM Agent and multimodal components."""
    
    def __init__(self, llm_agent: CoreLLMAgent, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multimodal LLM integration.
        
        Args:
            llm_agent: Core LLM Agent instance
            config: Optional configuration dictionary
        """
        self.llm_agent = llm_agent
        self.config = config or {}
        logger.info("Initialized multimodal LLM integration")
    
    def process_multimodal_input(self, processed_input: Dict[str, Any], 
                                context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process multimodal input with the Core LLM Agent.
        
        Args:
            processed_input: Processed multimodal input data
            context_data: Optional context data
            
        Returns:
            Processing result from the LLM
        """
        # Extract input type
        input_type = processed_input.get("input_type")
        
        if not input_type:
            error_msg = "Input type not specified in processed input"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Create a prompt based on the input type and content
        prompt = self._create_prompt_from_multimodal_input(processed_input, context_data)
        
        # Process with LLM
        llm_result = self.llm_agent.generate_response(prompt)
        
        return {
            "success": llm_result.get("success", False),
            "input_type": input_type,
            "response": llm_result.get("response", ""),
            "contains_math": self._check_for_math_content(llm_result.get("response", "")),
            "original_input": processed_input
        }
    
    def process_with_mathematical_result(self, processed_input: Dict[str, Any],
                                        math_result: Dict[str, Any],
                                        context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input with mathematical computation results.
        
        Args:
            processed_input: Processed multimodal input data
            math_result: Result from mathematical computation
            context_data: Optional context data
            
        Returns:
            Processing result from the LLM
        """
        # Create a prompt that includes the mathematical result
        prompt = self._create_prompt_with_math_result(processed_input, math_result, context_data)
        
        # Process with LLM
        llm_result = self.llm_agent.generate_response(prompt)
        
        return {
            "success": llm_result.get("success", False),
            "input_type": processed_input.get("input_type"),
            "response": llm_result.get("response", ""),
            "contains_math": True,
            "original_input": processed_input,
            "math_result": math_result
        }
    
    def generate_explanation(self, math_expression: str, 
                           computation_steps: List[Dict[str, Any]],
                           context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an explanation for mathematical steps.
        
        Args:
            math_expression: The mathematical expression
            computation_steps: List of computation steps
            context_data: Optional context data
            
        Returns:
            Explanation from the LLM
        """
        # Create a prompt for explanation
        prompt = self._create_explanation_prompt(math_expression, computation_steps, context_data)
        
        # Process with LLM
        llm_result = self.llm_agent.generate_response(prompt)
        
        return {
            "success": llm_result.get("success", False),
            "response": llm_result.get("response", ""),
            "expression": math_expression,
            "steps": computation_steps
        }
    
    def _create_prompt_from_multimodal_input(self, processed_input: Dict[str, Any],
                                          context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt from multimodal input.
        
        Args:
            processed_input: Processed multimodal input data
            context_data: Optional context data
            
        Returns:
            Prompt string for the LLM
        """
        input_type = processed_input.get("input_type")
        prompt_parts = []
        
        # Add context if available
        if context_data:
            conversation_history = context_data.get("conversation_history", [])
            if conversation_history:
                prompt_parts.append("Previous conversation:")
                for turn in conversation_history:
                    if "user" in turn:
                        prompt_parts.append(f"User: {turn['user']}")
                    if "assistant" in turn:
                        prompt_parts.append(f"Assistant: {turn['assistant']}")
                prompt_parts.append("")  # Empty line
        
        # Add system instruction
        prompt_parts.append("You are a mathematical assistant that can help with various mathematical problems. "
                         "You can solve equations, compute derivatives, integrals, and more. "
                         "Provide step-by-step explanations and be clear in your reasoning.")
        
        # Add input based on type
        if input_type == "text":
            text = processed_input.get("text", "")
            prompt_parts.append(f"User query: {text}")
            
        elif input_type == "image":
            # For OCR results
            recognized_latex = processed_input.get("recognized_latex", "")
            if recognized_latex:
                prompt_parts.append("The user has submitted a handwritten mathematical expression:")
                prompt_parts.append(f"LaTeX: {recognized_latex}")
                prompt_parts.append("Please interpret this expression and provide assistance.")
                
            # For diagrams
            if "diagrams" in processed_input:
                diagrams = processed_input.get("diagrams", [])
                prompt_parts.append("The user has submitted a diagram containing:")
                for i, diagram in enumerate(diagrams):
                    diagram_type = diagram.get("type", "unknown")
                    shape = diagram.get("shape", "unknown")
                    prompt_parts.append(f"- Diagram {i+1}: {diagram_type} {shape}")
                prompt_parts.append("Please interpret this diagram and provide assistance.")
                
            # For coordinate systems
            if "coordinates" in processed_input and processed_input.get("coordinates", {}).get("has_coordinates", False):
                prompt_parts.append("The user has submitted a coordinate system or graph.")
                prompt_parts.append("Please interpret this coordinate system and provide assistance.")
                
        elif input_type == "multipart":
            prompt_parts.append("The user has submitted multiple inputs:")
            
            parts = processed_input.get("parts", {})
            for key, part in parts.items():
                part_type = part.get("input_type")
                
                if part_type == "text":
                    text = part.get("text", "")
                    prompt_parts.append(f"- Text: {text}")
                    
                elif part_type == "image":
                    recognized_latex = part.get("recognized_latex", "")
                    if recognized_latex:
                        prompt_parts.append(f"- Handwritten expression: {recognized_latex}")
                        
            prompt_parts.append("Please interpret these inputs together and provide assistance.")
        
        # Add specific instruction if needed
        if context_data and "specific_instruction" in context_data:
            prompt_parts.append(f"Specific instruction: {context_data['specific_instruction']}")
        
        # Combine all parts
        return "\n\n".join(prompt_parts)
    
    def _create_prompt_with_math_result(self, processed_input: Dict[str, Any],
                                      math_result: Dict[str, Any],
                                      context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt that includes mathematical computation results.
        
        Args:
            processed_input: Processed multimodal input data
            math_result: Result from mathematical computation
            context_data: Optional context data
            
        Returns:
            Prompt string for the LLM
        """
        prompt_parts = []
        
        # Add context if available
        if context_data:
            conversation_history = context_data.get("conversation_history", [])
            if conversation_history:
                prompt_parts.append("Previous conversation:")
                for turn in conversation_history:
                    if "user" in turn:
                        prompt_parts.append(f"User: {turn['user']}")
                    if "assistant" in turn:
                        prompt_parts.append(f"Assistant: {turn['assistant']}")
                prompt_parts.append("")  # Empty line
        
        # Add system instruction
        prompt_parts.append("You are a mathematical assistant that can help with various mathematical problems. "
                         "Provide clear explanations based on the mathematical results.")
        
        # Add the original input
        input_type = processed_input.get("input_type")
        if input_type == "text":
            text = processed_input.get("text", "")
            prompt_parts.append(f"User query: {text}")
        elif input_type == "image":
            recognized_latex = processed_input.get("recognized_latex", "")
            prompt_parts.append(f"User input (handwritten): {recognized_latex}")
        
        # Add the mathematical result
        prompt_parts.append("\nMathematical computation result:")
        
        # Add different result types
        if "result" in math_result:
            prompt_parts.append(f"Result: {math_result['result']}")
            
        if "latex_result" in math_result:
            prompt_parts.append(f"LaTeX: {math_result['latex_result']}")
            
        if "steps" in math_result:
            prompt_parts.append("\nStep-by-step solution:")
            steps = math_result.get("steps", [])
            for i, step in enumerate(steps):
                prompt_parts.append(f"Step {i+1}: {step.get('description', '')}")
                if "latex" in step:
                    prompt_parts.append(f"   {step['latex']}")
        
        # Add instruction
        prompt_parts.append("\nPlease provide a clear explanation of this result in natural language. "
                         "Make sure to explain the reasoning behind each step and the final result.")
        
        # Combine all parts
        return "\n\n".join(prompt_parts)
    
    def _create_explanation_prompt(self, math_expression: str,
                                computation_steps: List[Dict[str, Any]],
                                context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for generating explanations.
        
        Args:
            math_expression: The mathematical expression
            computation_steps: List of computation steps
            context_data: Optional context data
            
        Returns:
            Prompt string for the LLM
        """
        prompt_parts = []
        
        # Add system instruction
        prompt_parts.append("You are a mathematical assistant that provides clear, educational explanations. "
                         "Your task is to explain the given mathematical computation in a way that is easy to understand.")
        
        # Add the expression
        prompt_parts.append(f"Mathematical expression: {math_expression}")
        
        # Add the steps
        prompt_parts.append("\nComputation steps:")
        for i, step in enumerate(computation_steps):
            prompt_parts.append(f"Step {i+1}: {step.get('description', '')}")
            if "latex" in step:
                prompt_parts.append(f"   {step['latex']}")
        
        # Add instruction
        prompt_parts.append("\nPlease provide a detailed explanation of these steps in natural language. "
                         "Your explanation should be educational and help someone understand the mathematical concepts involved. "
                         "Include explanations of any rules, theorems, or techniques being applied.")
        
        # Add specific requirements if available
        if context_data and "explanation_level" in context_data:
            level = context_data["explanation_level"]
            if level == "beginner":
                prompt_parts.append("The explanation should be suitable for beginners with minimal mathematical background.")
            elif level == "intermediate":
                prompt_parts.append("The explanation should be suitable for students with some mathematical background.")
            elif level == "advanced":
                prompt_parts.append("The explanation can include advanced mathematical concepts and terminology.")
        
        # Combine all parts
        return "\n\n".join(prompt_parts)
    
    def _check_for_math_content(self, text: str) -> bool:
        """
        Check if text contains mathematical content.
        
        Args:
            text: Text to check
            
        Returns:
            True if mathematical content is detected, False otherwise
        """
        # Simple heuristic for detecting mathematical content
        math_indicators = [
            "equation", "formula", "expression", "theorem",
            "=", "+", "-", "*", "/", "^", "\\frac", "\\int", "\\sum",
            "derivative", "integral", "matrix", "vector",
            "solve", "calculate", "compute"
        ]
        
        return any(indicator in text.lower() for indicator in math_indicators)

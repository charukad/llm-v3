"""
Integration between Mathematical Computation Agent and multimodal components.

This module handles the integration between the Mathematical Computation Agent
and the multimodal processing pipeline, allowing mathematical computation on
inputs from different modalities.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import re

from .math_agent import MathComputationAgent

logger = logging.getLogger(__name__)

class MathMultimodalIntegration:
    """Integration between Mathematical Computation Agent and multimodal components."""
    
    def __init__(self, math_agent: MathComputationAgent, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the math multimodal integration.
        
        Args:
            math_agent: Mathematical Computation Agent instance
            config: Optional configuration dictionary
        """
        self.math_agent = math_agent
        self.config = config or {}
        logger.info("Initialized math multimodal integration")
    
    def process_multimodal_input(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multimodal input with the Mathematical Computation Agent.
        
        Args:
            processed_input: Processed multimodal input data
            
        Returns:
            Processing result from the Mathematical Computation Agent
        """
        # Extract input type and relevant data
        input_type = processed_input.get("input_type")
        
        if not input_type:
            error_msg = "Input type not specified in processed input"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Extract the mathematical expression based on input type
        expressions = self._extract_expressions_from_input(processed_input)
        
        if not expressions:
            return {
                "success": False,
                "error": "No mathematical expressions found in input"
            }
        
        # Process each expression
        results = []
        for expr in expressions:
            # Determine the operation based on the expression and input
            operation = self._determine_operation(expr, processed_input)
            
            # Process with math agent
            try:
                result = self.math_agent.process_expression(expr, operation)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing expression '{expr}': {str(e)}")
                results.append({
                    "success": False,
                    "expression": expr,
                    "error": str(e)
                })
        
        # Combine results
        if len(results) == 1:
            combined_result = results[0]
        else:
            combined_result = {
                "success": any(r.get("success", False) for r in results),
                "results": results,
                "expression_count": len(results)
            }
        
        # Add original input data
        combined_result["original_input"] = processed_input
        combined_result["input_type"] = input_type
        
        return combined_result
    
    def process_expression_with_operation(self, expression: str, operation: str,
                                        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a specific expression with a given operation.
        
        Args:
            expression: Mathematical expression
            operation: Operation to perform (e.g., "solve", "differentiate")
            parameters: Optional parameters for the operation
            
        Returns:
            Processing result from the Mathematical Computation Agent
        """
        # Process with math agent
        try:
            result = self.math_agent.process_expression(expression, operation, parameters)
            return result
        except Exception as e:
            logger.error(f"Error processing expression '{expression}' with operation '{operation}': {str(e)}")
            return {
                "success": False,
                "expression": expression,
                "operation": operation,
                "error": str(e)
            }
    
    def _extract_expressions_from_input(self, processed_input: Dict[str, Any]) -> List[str]:
        """
        Extract mathematical expressions from processed input.
        
        Args:
            processed_input: Processed multimodal input data
            
        Returns:
            List of mathematical expressions
        """
        input_type = processed_input.get("input_type")
        expressions = []
        
        if input_type == "text":
            # For text, try to extract expressions from the text
            text = processed_input.get("text", "")
            text_type = processed_input.get("text_type", "")
            
            if text_type == "latex":
                # Already LaTeX, just add directly
                expressions.append(text)
            else:
                # Extract any math-like expressions
                # This is a simplified approach; a real implementation would be more sophisticated
                extracted = self._extract_math_from_text(text)
                expressions.extend(extracted)
                
        elif input_type == "image":
            # For images with OCR results
            recognized_latex = processed_input.get("recognized_latex", "")
            if recognized_latex:
                expressions.append(recognized_latex)
                
        elif input_type == "multipart":
            # For multipart input, process each part
            parts = processed_input.get("parts", {})
            for key, part in parts.items():
                part_expressions = self._extract_expressions_from_input(part)
                expressions.extend(part_expressions)
        
        return expressions
    
    def _extract_math_from_text(self, text: str) -> List[str]:
        """
        Extract mathematical expressions from text.
        
        Args:
            text: Text to extract expressions from
            
        Returns:
            List of extracted expressions
        """
        # This is a simplified implementation; a real one would be more sophisticated
        
        # Look for explicit LaTeX delimiters
        latex_pattern = r'\$(.*?)\$'
        latex_matches = re.findall(latex_pattern, text)
        
        # Look for equation-like patterns
        equation_pattern = r'([a-zA-Z0-9]+\s*=\s*[^=.!?;]*)'
        equation_matches = re.findall(equation_pattern, text)
        
        # Look for expressions with operators
        expression_pattern = r'([a-zA-Z0-9]+\s*[\+\-\*\/\^\(\)]\s*[^=.!?;]*)'
        expression_matches = re.findall(expression_pattern, text)
        
        # Combine all matches
        all_matches = latex_matches + equation_matches + expression_matches
        
        # Remove duplicates and clean up
        unique_matches = []
        for match in all_matches:
            cleaned = match.strip()
            if cleaned and cleaned not in unique_matches:
                unique_matches.append(cleaned)
        
        return unique_matches
    
    def _determine_operation(self, expression: str, processed_input: Dict[str, Any]) -> str:
        """
        Determine the appropriate operation for an expression.
        
        Args:
            expression: Mathematical expression
            processed_input: Processed multimodal input data
            
        Returns:
            Operation string
        """
        # Extract the text to look for operation hints
        text = ""
        input_type = processed_input.get("input_type")
        
        if input_type == "text":
            text = processed_input.get("text", "")
        elif input_type == "multipart":
            # Look for text parts
            parts = processed_input.get("parts", {})
            for key, part in parts.items():
                if part.get("input_type") == "text":
                    text += part.get("text", "") + " "
        
        text = text.lower()
        
        # Check for specific operation keywords
        if re.search(r'\b(solve|find|determine|calculate|compute)\b', text) and \
           re.search(r'\b(equation|equals|=)\b', text):
            return "solve"
        
        elif re.search(r'\b(differentiate|derivative|derive)\b', text):
            return "differentiate"
        
        elif re.search(r'\b(integrate|integral|antiderivative)\b', text):
            return "integrate"
        
        elif re.search(r'\b(simplify|simplification)\b', text):
            return "simplify"
        
        elif re.search(r'\b(expand|expansion)\b', text):
            return "expand"
        
        elif re.search(r'\b(factor|factorize|factorization)\b', text):
            return "factor"
        
        elif re.search(r'\b(limit)\b', text):
            return "limit"
        
        # Default operation based on expression structure
        if "=" in expression:
            return "solve"
        elif any(op in expression for op in ["+", "-", "*", "/", "^"]):
            return "evaluate"
        else:
            return "simplify"  # Default operation

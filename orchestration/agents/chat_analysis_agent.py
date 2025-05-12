"""
Chat Analysis Agent for identifying plot requests from natural language.

This agent processes natural language input to determine if it's a request for
a visualization and extracts the necessary parameters to create the plot.
"""
import json
import re
import logging
import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
from datetime import datetime

from ..message_bus.message_formats import Message, MessageType, MessagePriority, create_message
from ..monitoring.logger import get_logger
from orchestration.agents.base_agent import BaseAgent

from core.agent.llm_agent import CoreLLMAgent

logger = get_logger(__name__)

# Mathematical concept library for common functions
MATH_CONCEPT_LIBRARY = {
    # Standard named functions
    "mexican hat": "(x**2 + y**2 - 1)**2",
    "mexican hat potential": "(x**2 + y**2 - 1)**2",
    "sombrero": "(x**2 + y**2 - 1)**2",
    
    # Wave and oscillation patterns
    "ripple": "sin(sqrt(x**2 + y**2))",
    "damped ripple": "exp(-0.1*sqrt(x**2+y**2)) * sin(sqrt(x**2+y**2))",
    "wave interference": "sin(x) * sin(y)",
    "standing wave": "sin(x) * sin(y)",
    
    # Physics and mathematical models
    "gaussian": "exp(-(x**2 + y**2))",
    "normal distribution": "exp(-(x**2 + y**2)/2) / (2*pi)",
    "bessel function": "sin(sqrt(x**2 + y**2)) / sqrt(x**2 + y**2)",
    "harmonic oscillator": "0.5 * (x**2 + y**2)",
    "quantum harmonic oscillator": "0.5 * (x**2 + y**2)",
    "double well potential": "(x**2 - 1)**2 + y**2",
    "saddle point": "x**2 - y**2",
    "monkey saddle": "x**3 - 3*x*y**2",
    
    # Other interesting surfaces
    "hyperbolic paraboloid": "x**2 - y**2",
    "elliptic paraboloid": "x**2 + y**2",
    "hyperboloid": "x**2 + y**2 - z**2",
    "cone": "sqrt(x**2 + y**2)",
    "sinc function": "sin(sqrt(x**2 + y**2)) / (sqrt(x**2 + y**2) + 0.001)",
    "ripple tank": "sin(2*sqrt(x**2 + y**2)) * exp(-0.1*sqrt(x**2 + y**2))",
}

# System prompt for plot extraction
PLOT_EXTRACTION_PROMPT = """
You are a specialized assistant for analyzing mathematical visualization requests.
Your task is to identify if the user is requesting a plot or visualization and extract all necessary information to create it.

IMPORTANT: When extracting mathematical expressions, preserve them EXACTLY as stated by the user. Do not simplify or modify expressions.
Even complex expressions like "(x^7+y^2)-3" or "sin(x)*cos(y)/log(x+y)" should be preserved exactly.

For each request, determine:
1. Is this a visualization request? (yes/no)
2. What type of plot is needed? (function_2d, functions_2d, function_3d, parametric_3d)
3. Extract all relevant parameters based on the plot type:

For function_2d:
- Mathematical expression(s) - PRESERVE EXACTLY as stated
- X range (min and max values)
- Plot title (optional)
- Axis labels (optional)

For functions_2d:
- List of mathematical expressions - PRESERVE EXACTLY as stated
- List of labels for each expression (optional)
- X range (min and max values)
- Plot title (optional)
- Axis labels (optional)

For function_3d:
- Mathematical expression for z=f(x,y) - PRESERVE EXACTLY as stated
- X range and Y range
- Plot title (optional)
- Axis labels (optional)
- View angle (optional)

For parametric_3d:
- X, Y, and Z expressions in terms of parameter t - PRESERVE EXACTLY as stated
- T range (min and max values)
- Plot title (optional)
- Axis labels (optional)
- Color (optional)

Examples:
- "Plot sin(x)" -> function_2d with expression="sin(x)"
- "Create a 3D visualization of (x^7+y^2)-3" -> function_3d with expression="(x^7+y^2)-3"
- "Show me the plot of x^2*exp(-x)" -> function_2d with expression="x^2*exp(-x)"

Return your analysis as a structured JSON object with these fields.
Return null values for any information that isn't specified in the request.
"""

class ChatAnalysisAgent(BaseAgent):
    """
    Agent for analyzing chat messages to identify plot requests and extract parameters.
    """
    
    def __init__(self, agent_id: str = "chat_analysis_agent"):
        """
        Initialize the Chat Analysis Agent.
        
        Args:
            agent_id: Unique agent identifier
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="text_processing",
            capabilities=["identify_plot_request", "extract_plot_parameters", "route_visualization_requests"],
            description="Agent for analyzing chat messages to identify plot requests"
        )
        
        # Reference to LLM agent for processing
        self.llm_agent = None
        self.system_prompt = PLOT_EXTRACTION_PROMPT
        
    async def initialize(self):
        """Initialize specialized components."""
        # Try to get the LLM agent instance
        try:
            from api.rest.server import get_core_llm_agent
            self.llm_agent = get_core_llm_agent()
            if not self.llm_agent:
                logger.warning("CoreLLMAgent not found during initialization")
        except Exception as e:
            logger.error(f"Error initializing LLM agent: {str(e)}")
        
        await super().initialize()
    
    def register_message_handlers(self):
        """Register message handlers for this agent."""
        self.register_message_handler(
            MessageType.REQUEST,
            self._handle_request
        )
        
    async def _handle_request(self, message: Message):
        """
        Handle incoming request messages.
        
        Args:
            message: Incoming message
        """
        # Extract message body
        body = message.body
        
        # Process the request
        result = self.process_message(body)
        
        # Create response message
        response = create_message(
            sender=self.agent_id,
            recipient=message.header.sender,
            message_type=MessageType.RESPONSE,
            correlation_id=message.header.message_id,
            body=result
        )
        
        # Send the response
        await self.message_bus.send_message(response)
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message to identify plot requests and extract parameters.
        
        Args:
            message: The message containing the text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract text content
            text = message.get("text", "")
            if not text:
                text = message.get("content", "")
                
            if not text:
                return {
                    "success": False,
                    "error": "No text content found in message"
                }
            
            # Analyze the text for plot requests
            analysis = self.analyze_plot_request(text)
            
            # Add original text to the response
            analysis["original_text"] = text
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Error processing message: {str(e)}",
                "original_text": message.get("text", message.get("content", ""))
            }
    
    def extract_expression(self, text: str) -> str:
        """
        Extract a mathematical expression from text.
        
        Args:
            text: Text to extract expression from
            
        Returns:
            Extracted expression or None if no expression found
        """
        # Check for conditional expressions first
        conditional_expr = self.handle_conditional_expression(text)
        if conditional_expr:
            return conditional_expr
        
        # Look up mathematical concepts
        concept_formula = self.lookup_concept(text)
        if concept_formula:
            return concept_formula
        
        # Check for expressions enclosed in quotes (most reliable)
        quoted_expr = re.search(r'["\']([^"\']+)["\']', text, re.IGNORECASE)
        if quoted_expr:
            return quoted_expr.group(1).strip()
        
        # Check for complex expressions using advanced patterns
        complex_patterns = [
            r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)(?:from|with|where|for)',  # expression between keywords
            r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)$',  # expression at end of sentence
            r'([a-z0-9\^\(\)\+\-\*\/\.\s]+)\s*(?:from|with|between)\s+[-\d]',  # expression before range
            r'z\s*=\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # z = expression
            r'f\s*\(\s*x\s*(?:,\s*y)?\s*\)\s*=\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # f(x) = or f(x,y) = 
            r'y\s*=\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)'  # y = expression
        ]
        
        for pattern in complex_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                expr = match.group(1).strip()
                # Clean up the expression
                return re.sub(r'\s+', '', expr)  # Remove all whitespace from expression
        
        # Define original patterns (as fallback)
        expression_patterns = [
            r'sin\s*\(\s*x\s*\)',  # sin(x)
            r'cos\s*\(\s*x\s*\)',  # cos(x)
            r'tan\s*\(\s*x\s*\)',  # tan(x)
            r'x\s*\^\s*\d+',       # x^2, x^3, etc.
            r'e\s*\^\s*x',         # e^x
            r'log\s*\(\s*x\s*\)',  # log(x)
            r'ln\s*\(\s*x\s*\)',   # ln(x)
        ]
        
        # Try each pattern
        for pattern in expression_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        # Check for common expression keywords
        expression_keywords = [
            "plot", "graph", "function", "expression", "formula", "equation"
        ]
        
        # Look for expressions in the vicinity of these keywords
        for keyword in expression_keywords:
            keyword_pos = text.lower().find(keyword)
            if keyword_pos != -1:
                # Extract the text after the keyword
                after_keyword = text[keyword_pos + len(keyword):].strip()
                # Look for a potential expression
                expr_match = re.search(r'[a-z0-9\^\(\)\+\-\*\/\.]+', after_keyword)
                if expr_match:
                    return expr_match.group(0).strip()
        
        # Fallback heuristics
        if "sin" in text.lower():
            return "sin(x)"
        elif "cos" in text.lower():
            return "cos(x)"
        elif "plot" in text.lower():
            return "x^2"  # Default fallback
        
        return None
    
    def extract_multiple_expressions(self, text: str) -> List[str]:
        """
        Extract multiple expressions from text.
        
        Args:
            text: Text to extract expressions from
            
        Returns:
            List of extracted expressions
        """
        expressions = []
        
        # Look for sin(x) and cos(x) pattern which is common
        if "sin" in text.lower() and "cos" in text.lower():
            expressions = ["sin(x)", "cos(x)"]
            return expressions
        
        # Individual pattern matching
        patterns = [
            r'sin\s*\(\s*x\s*\)',  # sin(x)
            r'cos\s*\(\s*x\s*\)',  # cos(x)
            r'tan\s*\(\s*x\s*\)',  # tan(x) 
            r'x\s*\^\s*\d+',       # x^2, x^3, etc.
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                expression = match.group(0).strip()
                if expression not in expressions:
                    expressions.append(expression)
        
        return expressions
    
    def extract_3d_expression(self, text: str) -> str:
        """
        Extract a 3D function expression from text.
        
        Args:
            text: Text to extract expression from
            
        Returns:
            Extracted expression or None if no expression found
        """
        # Check for conditional expressions first
        conditional_expr = self.handle_conditional_expression(text)
        if conditional_expr:
            return conditional_expr
        
        # Look up mathematical concepts
        concept_formula = self.lookup_concept(text)
        if concept_formula:
            return concept_formula
        
        # Check for expressions enclosed in quotes first (most reliable)
        quoted_expr = re.search(r'["\']([^"\']+)["\']', text, re.IGNORECASE)
        if quoted_expr:
            return quoted_expr.group(1).strip()
        
        # Advanced patterns for 3D expressions
        advanced_patterns = [
            r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)(?:from|with|where|for)',  # expression between keywords
            r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)$',  # expression at end of sentence
            r'3d\s+(?:plot|graph|visualization|surface)\s+(?:of|for)?\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # 3D plot of ...
            r'surface\s+(?:plot|graph)?\s+(?:of|for)?\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # surface plot of ...
        ]
        
        for pattern in advanced_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                expr = match.group(1).strip()
                # Clean up the expression
                return re.sub(r'\s+', '', expr)  # Remove all whitespace from expression
        
        # Check for common 3D function pattern: z = f(x,y)
        z_pattern = r'z\s*=\s*([^\n]+)'
        match = re.search(z_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Check for specific 3D function patterns
        patterns = [
            r'sin\s*\(\s*x\s*\)\s*\*\s*cos\s*\(\s*y\s*\)',  # sin(x)*cos(y)
            r'x\s*\^\s*2\s*\+\s*y\s*\^\s*2',                # x^2 + y^2
            r'x\s*\^\s*\d+\s*\+\s*y\s*\^\s*\d+',            # x^n + y^m
            r'([a-z0-9\^\(\)\+\-\*\/\.]+)',                 # Generic expression pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        # Check for 3D visualization with sin and cos
        if "3d" in text.lower() and "sin" in text.lower() and "cos" in text.lower():
            return "sin(x)*cos(y)"
        
        # Default 3D function
        return "x^2 + y^2"
    
    def extract_parametric_3d(self, text: str) -> Dict[str, str]:
        """
        Extract parametric 3D curve expressions from text.
        
        Args:
            text: Text to extract expressions from
            
        Returns:
            Dictionary with x, y, z expressions or None if not found
        """
        # Check for specific pattern for helix
        if "helix" in text.lower():
            return {
                "x_expression": "cos(t)",
                "y_expression": "sin(t)",
                "z_expression": "t"
            }
        
        # Try to extract individual components
        x_pattern = r'x\s*=\s*([^\n,;]+)'
        y_pattern = r'y\s*=\s*([^\n,;]+)'
        z_pattern = r'z\s*=\s*([^\n,;]+)'
        
        expressions = {}
        
        match = re.search(x_pattern, text, re.IGNORECASE)
        if match:
            expressions["x_expression"] = match.group(1).strip()
            
        match = re.search(y_pattern, text, re.IGNORECASE)
        if match:
            expressions["y_expression"] = match.group(1).strip()
            
        match = re.search(z_pattern, text, re.IGNORECASE)
        if match:
            expressions["z_expression"] = match.group(1).strip()
        
        if len(expressions) == 3:  # All three components found
            return expressions
        
        # If not all components found but we have "parametric" and 3D
        if "parametric" in text.lower() and "3d" in text.lower():
            # Default parametric 3D curve (helix)
            return {
                "x_expression": "cos(t)",
                "y_expression": "sin(t)",
                "z_expression": "t"
            }
        
        return expressions
    
    def sanitize_expression(self, expression: str) -> str:
        """
        Sanitize and standardize a mathematical expression.
        
        Args:
            expression: The raw expression string
            
        Returns:
            Sanitized expression
        """
        if not expression:
            return None
        
        # Check if it's already a numpy where expression
        if expression.startswith("np.where"):
            return expression
        
        # Replace caret notation with double asterisk for Python
        expression = re.sub(r'(\w+)\s*\^\s*(\d+)', r'\1**\2', expression)
        
        # Replace any remaining ^ with ** for Python
        expression = expression.replace('^', '**')
        
        # Handle special functions and constants
        expression = (expression
                     .replace('pi', 'np.pi')
                     .replace('sin(', 'np.sin(')
                     .replace('cos(', 'np.cos(')
                     .replace('tan(', 'np.tan(')
                     .replace('exp(', 'np.exp(')
                     .replace('log(', 'np.log(')
                     .replace('sqrt(', 'np.sqrt(')
                     .replace('abs(', 'np.abs('))
        
        # If the expression contains 'np.' already, don't modify it again
        if 'np.' not in expression:
            # Standardize the expression for evaluation
            # Add explicit multiplication where implied
            expression = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expression)
            expression = re.sub(r'(\))([a-zA-Z\(])', r'\1*\2', expression)
        
        # Remove all whitespace
        expression = re.sub(r'\s+', '', expression)
        
        return expression
    
    def analyze_plot_request(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to identify plot requests and extract parameters.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Add numpy import instruction to ensure conditional expressions work
        numpy_import_instruction = "import numpy as np"
        
        # Try direct extraction of complex expressions first
        expression_indicators = [
            "x^", "sin(", "cos(", "log(", "exp(", "(x", "z=", "f(x", "y="
        ]
        
        # Look for mathematical concepts first
        concept_formula = self.lookup_concept(text)
        if concept_formula:
            logger.info(f"Found mathematical concept formula: {concept_formula}")
            sanitized_formula = self.sanitize_expression(concept_formula)
            return {
                "success": True,
                "is_visualization_request": True,
                "plot_type": "function_3d" if "y" in concept_formula else "function_2d",
                "parameters": {
                    "expression": sanitized_formula,
                    "numpy_import": numpy_import_instruction
                },
                "analysis": {
                    "is_visualization_request": True,
                    "plot_type": "function_3d" if "y" in concept_formula else "function_2d",
                    "mathematical_expression": concept_formula,
                    "concept_match": True
                }
            }
        
        # Check for conditional expressions
        conditional_expr = self.handle_conditional_expression(text)
        if conditional_expr:
            logger.info(f"Found conditional expression: {conditional_expr}")
            return {
                "success": True,
                "is_visualization_request": True,
                "plot_type": "function_3d" if "y" in text.lower() else "function_2d",
                "parameters": {
                    "expression": conditional_expr,
                    "numpy_import": numpy_import_instruction
                },
                "analysis": {
                    "is_visualization_request": True,
                    "plot_type": "function_3d" if "y" in text.lower() else "function_2d",
                    "mathematical_expression": conditional_expr,
                    "is_conditional": True
                }
            }
        
        # Direct extraction for 3D plots
        if "3d" in text.lower() and any(indicator in text.lower() for indicator in expression_indicators):
            logger.info("Attempting direct extraction for 3D plot")
            direct_extraction = True
            extracted_expr = self.extract_3d_expression(text)
            if extracted_expr:
                return {
                    "success": True,
                    "is_visualization_request": True,
                    "plot_type": "function_3d",
                    "parameters": {
                        "expression": self.sanitize_expression(extracted_expr),
                        "numpy_import": numpy_import_instruction
                    },
                    "analysis": {
                        "is_visualization_request": True,
                        "plot_type": "function_3d",
                        "mathematical_expression": extracted_expr
                    }
                }
        
        # Direct extraction for 2D plots
        elif any(indicator in text.lower() for indicator in expression_indicators):
            logger.info("Attempting direct extraction for 2D plot")
            direct_extraction = True
            extracted_expr = self.extract_expression(text)
            if extracted_expr:
                return {
                    "success": True,
                    "is_visualization_request": True,
                    "plot_type": "function_2d",
                    "parameters": {
                        "expression": self.sanitize_expression(extracted_expr)
                    },
                    "analysis": {
                        "is_visualization_request": True,
                        "plot_type": "function_2d",
                        "mathematical_expression": extracted_expr
                    }
                }
        
        # Special case for helix which has a specific structure
        if "helix" in text.lower() and all(term in text.lower() for term in ["x", "y", "z", "cos", "sin", "t"]):
            return {
                "success": True,
                "is_visualization_request": True,
                "plot_type": "parametric_3d",
                "parameters": {
                    "x_expression": "cos(t)",
                    "y_expression": "sin(t)",
                    "z_expression": "t",
                    "t_min": 0,
                    "t_max": 6.28,
                    "title": "3D Helix"
                },
                "analysis": {
                    "is_visualization_request": True,
                    "plot_type": "parametric_3d"
                }
            }
                
        # Continue with LLM analysis if direct extraction failed or wasn't attempted
        # Ensure we have the LLM agent
        if not self.llm_agent:
            try:
                from api.rest.server import get_core_llm_agent
                self.llm_agent = get_core_llm_agent()
                if not self.llm_agent:
                    return {
                        "success": False,
                        "error": "LLM agent not available for text analysis"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Could not access LLM agent: {str(e)}"
                }
        
        try:
            # Use the LLM to analyze the request
            llm_response = self.llm_agent.generate_response(
                prompt=text,
                system_prompt=self.system_prompt,
                temperature=0.2
            )
            
            if not llm_response.get("success", False):
                return {
                    "success": False,
                    "error": f"LLM analysis failed: {llm_response.get('error', 'Unknown error')}"
                }
            
            response_text = llm_response.get("response", "")
            
            # Extract JSON from the response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'(\{[\s\S]*\})', response_text)
                if json_match:
                    json_text = json_match.group(1).strip()
                else:
                    # Just use the full response text and hope for the best
                    json_text = response_text
            
            # Try to clean up the JSON text to make it more parseable
            # Remove any explanatory text before or after the JSON
            json_text = re.sub(r'^[^{]*', '', json_text)
            json_text = re.sub(r'[^}]*$', '', json_text)
            
            try:
                analysis = json.loads(json_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response: {response_text}")
                
                # Manual fallback detection for common visualization keywords
                if any(keyword in text.lower() for keyword in ["plot", "graph", "visualize", "chart", "draw", "function"]):
                    # Determine if this is 2D or 3D
                    is_3d = "3d" in text.lower() or "3-d" in text.lower() or "three dimension" in text.lower()
                    
                    # Create a basic analysis structure
                    analysis = {
                        "is_visualization_request": True,
                        "plot_type": "function_3d" if is_3d else "function_2d"
                    }
                    
                    # Try direct extraction if this is likely a visualization
                    extracted_expr = None
                    if is_3d:
                        extracted_expr = self.extract_3d_expression(text)
                    else:
                        extracted_expr = self.extract_expression(text)
                        
                    if extracted_expr:
                        # Use sanitized version for parameters
                        analysis["parameters"] = {
                            "expression": self.sanitize_expression(extracted_expr)
                        }
                        # Save original expression for display/reference
                        analysis["mathematical_expression"] = extracted_expr
                else:
                    # Return a simple structure with the raw text
                    return {
                        "success": True,
                        "is_visualization_request": False,
                        "raw_llm_response": response_text
                    }
            
            # Determine if this is a visualization request
            is_visualization = analysis.get("is_visualization_request", False)
            if isinstance(is_visualization, str):
                is_visualization = is_visualization.lower() == "yes"
            
            # Fallback detection if LLM didn't identify it correctly
            if not is_visualization:
                # Check for common visualization keywords
                keywords = ["plot", "graph", "visualize", "chart", "draw", "show me", "display"]
                if any(keyword in text.lower() for keyword in keywords):
                    is_visualization = True
                    analysis["is_visualization_request"] = True
                    # Default to function_2d if no plot type
                    if "plot_type" not in analysis:
                        analysis["plot_type"] = "function_2d"
                    # Initialize parameters if not present
                    if "parameters" not in analysis:
                        analysis["parameters"] = {}
            
            if not is_visualization:
                return {
                    "success": True,
                    "is_visualization_request": False,
                    "plot_type": None,
                    "parameters": {},
                    "analysis": analysis
                }
            
            # Extract the plot type and parameters
            plot_type = analysis.get("plot_type", "function_2d")
            if isinstance(plot_type, str):
                plot_type = plot_type.strip()  # Remove leading/trailing spaces
            parameters = analysis.get("parameters", {})
            
            # Fallback extraction if parameters are incomplete
            if plot_type == "function_2d" and "expression" not in parameters:
                extracted_expr = self.extract_expression(text)
                if extracted_expr:
                    parameters["expression"] = self.sanitize_expression(extracted_expr)
                    # Save original expression for display/reference
                    analysis["mathematical_expression"] = extracted_expr
            
            elif plot_type == "functions_2d":
                if "expressions" not in parameters:
                    # Try to extract multiple expressions
                    expressions = self.extract_multiple_expressions(text)
                    if expressions:
                        # Sanitize all expressions
                        parameters["expressions"] = [self.sanitize_expression(expr) for expr in expressions]
                        # Save original expressions for display/reference
                        analysis["mathematical_expressions"] = expressions
                    elif "expression" in parameters:
                        # Convert single expression to a list
                        parameters["expressions"] = [parameters["expression"]]
            
            elif plot_type == "function_3d" and "expression" not in parameters:
                extracted_expr = self.extract_3d_expression(text)
                if extracted_expr:
                    parameters["expression"] = self.sanitize_expression(extracted_expr)
                    # Save original expression for display/reference
                    analysis["mathematical_expression"] = extracted_expr
            
            elif plot_type == "parametric_3d":
                if not all(k in parameters for k in ["x_expression", "y_expression", "z_expression"]):
                    parametric_exprs = self.extract_parametric_3d(text)
                    if parametric_exprs:
                        # Sanitize all expressions
                        for key, expr in parametric_exprs.items():
                            parametric_exprs[key] = self.sanitize_expression(expr)
                        parameters.update(parametric_exprs)
            
            # Ensure we have the minimal required parameters for each plot type
            if plot_type == "function_2d":
                if "expression" not in parameters:
                    return {
                        "success": False,
                        "error": "Missing required parameter 'expression' for function_2d plot",
                        "is_visualization_request": True,
                        "plot_type": plot_type,
                        "parameters": parameters
                    }
            
            elif plot_type == "functions_2d":
                if "expressions" not in parameters:
                    # Try to extract multiple expressions
                    if "expression" in parameters:
                        # Convert single expression to a list
                        parameters["expressions"] = [parameters["expression"]]
                    else:
                        return {
                            "success": False,
                            "error": "Missing required parameter 'expressions' for functions_2d plot",
                            "is_visualization_request": True,
                            "plot_type": plot_type,
                            "parameters": parameters
                        }
            
            elif plot_type == "function_3d":
                if "expression" not in parameters:
                    return {
                        "success": False,
                        "error": "Missing required parameter 'expression' for function_3d plot",
                        "is_visualization_request": True,
                        "plot_type": plot_type,
                        "parameters": parameters
                    }
            
            elif plot_type == "parametric_3d":
                if not all(k in parameters for k in ["x_expression", "y_expression", "z_expression"]):
                    return {
                        "success": False,
                        "error": "Missing required parameters 'x_expression', 'y_expression', 'z_expression' for parametric_3d plot",
                        "is_visualization_request": True,
                        "plot_type": plot_type,
                        "parameters": parameters
                    }
            
            # Format the return object
            return {
                "success": True,
                "is_visualization_request": True,
                "plot_type": plot_type,
                "parameters": parameters,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in plot request analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Error analyzing plot request: {str(e)}"
            }
    
    async def process_visualization_request(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a visualization request based on the analysis.
        
        Args:
            analysis_result: Analysis result from analyze_plot_request
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing visualization request with type: {analysis_result.get('plot_type')}")
        
        if not analysis_result.get("success", False) or not analysis_result.get("is_visualization_request", False):
            logger.warning("Not a valid visualization request")
            return {
                "success": False,
                "error": "Not a valid visualization request"
            }
        
        try:
            # Extract plot type and parameters
            plot_type = analysis_result.get("plot_type")
            parameters = analysis_result.get("parameters", {})
            
            logger.info(f"Visualization parameters: {parameters}")
            
            if not plot_type:
                logger.error("No plot type specified in analysis result")
                return {
                    "success": False,
                    "error": "No plot type specified in analysis result"
                }
            
            # Route to the appropriate visualization endpoint
            # Imports inside function to avoid circular imports
            logger.info(f"Importing visualization endpoints for plot type: {plot_type}")
            from api.rest.routes.visualization import (
                plot_function, plot_multiple_functions, plot_3d_surface, plot_parametric_3d
            )
            
            if plot_type == "function_2d":
                logger.info("Processing function_2d visualization")
                return await plot_function(**parameters)
            
            elif plot_type == "functions_2d":
                logger.info("Processing functions_2d visualization")
                return await plot_multiple_functions(**parameters)
            
            elif plot_type == "function_3d":
                logger.info("Processing function_3d visualization")
                return await plot_3d_surface(**parameters)
            
            elif plot_type == "parametric_3d":
                logger.info("Processing parametric_3d visualization")
                return await plot_parametric_3d(**parameters)
            
            else:
                logger.error(f"Unsupported plot type: {plot_type}")
                return {
                    "success": False,
                    "error": f"Unsupported plot type: {plot_type}"
                }
                
        except Exception as e:
            logger.error(f"Error processing visualization request: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Error processing visualization request: {str(e)}"
            }

# Create a singleton instance
_chat_analysis_agent_instance = None

def get_chat_analysis_agent() -> ChatAnalysisAgent:
    """Get or create the chat analysis agent singleton instance."""
    global _chat_analysis_agent_instance
    if _chat_analysis_agent_instance is None:
        _chat_analysis_agent_instance = ChatAnalysisAgent()
    return _chat_analysis_agent_instance 

# Implement conditional expression handler
def handle_conditional_expression(self, expression_text: str) -> str:
    """
    Parse and handle conditional expressions (piecewise functions).
    
    Args:
        expression_text: Text containing the conditional expression
        
    Returns:
        Processed expression using numpy.where() for conditionals
    """
    # Check for the keyword "where" or "if", which often indicate conditions
    if any(keyword in expression_text.lower() for keyword in [" where ", " if ", " for ", " when "]):
        try:
            # Common patterns for conditional expressions
            # Pattern: "expr1 for condition, expr2 otherwise"
            pattern1 = r'(.*?)\s+(?:for|where|if|when)\s+(.*?),\s*(.*?)\s+(?:otherwise|else)'
            # Pattern: "expr1 for condition, otherwise expr2"
            pattern2 = r'(.*?)\s+(?:for|where|if|when)\s+(.*?),\s*(?:otherwise|else)\s+(.*)'
            # Pattern: "If condition then expr1 else expr2"
            pattern3 = r'(?:if|when)\s+(.*?)\s+then\s+(.*?)\s+else\s+(.*)'
            
            match = None
            expr1, condition, expr2 = None, None, None
            
            for pattern in [pattern1, pattern2, pattern3]:
                match = re.search(pattern, expression_text, re.IGNORECASE)
                if match:
                    if pattern == pattern3:  # Different group order for pattern3
                        condition = match.group(1).strip()
                        expr1 = match.group(2).strip()
                        expr2 = match.group(3).strip()
                    else:
                        expr1 = match.group(1).strip()
                        condition = match.group(2).strip()
                        expr2 = match.group(3).strip()
                    break
            
            if match:
                # Clean up expressions and condition
                expr1 = self.sanitize_expression(expr1)
                expr2 = self.sanitize_expression(expr2)
                
                # Translate condition to Python syntax
                condition = (condition
                            .replace("^", "**")
                            .replace("=", "==")
                            .replace("<>", "!=")
                            .replace(" and ", " & ")
                            .replace(" or ", " | "))
                
                # Construct numpy.where() expression
                numpy_expr = f"np.where({condition}, {expr1}, {expr2})"
                logger.info(f"Constructed conditional expression: {numpy_expr}")
                return numpy_expr
        except Exception as e:
            logger.warning(f"Failed to parse conditional expression: {str(e)}")
    
    return None

# Add a method to handle concept lookup
def lookup_concept(self, text: str) -> str:
    """
    Look for mathematical concept terms in the text and return the matching formula.
    
    Args:
        text: The text to search for mathematical concepts
        
    Returns:
        The mathematical formula if a concept is found, None otherwise
    """
    text_lower = text.lower()
    
    # First look for exact phrases
    for concept, formula in MATH_CONCEPT_LIBRARY.items():
        if concept in text_lower:
            logger.info(f"Found mathematical concept: {concept} -> {formula}")
            return formula
    
    # Then look for partial matches (for more flexibility)
    for concept, formula in MATH_CONCEPT_LIBRARY.items():
        # Split concept into words and check if all words are in the text
        concept_words = concept.split()
        if len(concept_words) > 1 and all(word in text_lower for word in concept_words):
            logger.info(f"Found partial match for concept: {concept} -> {formula}")
            return formula
    
    return None

# Enhance the extract_expression method to use the concept library
def extract_expression(self, text: str) -> str:
    """
    Extract a mathematical expression from text.
    
    Args:
        text: Text to extract expression from
        
    Returns:
        Extracted expression or None if no expression found
    """
    # Check for conditional expressions first
    conditional_expr = self.handle_conditional_expression(text)
    if conditional_expr:
        return conditional_expr
    
    # Look up mathematical concepts
    concept_formula = self.lookup_concept(text)
    if concept_formula:
        return concept_formula
    
    # Check for expressions enclosed in quotes (most reliable)
    quoted_expr = re.search(r'["\']([^"\']+)["\']', text, re.IGNORECASE)
    if quoted_expr:
        return quoted_expr.group(1).strip()
    
    # Check for complex expressions using advanced patterns
    complex_patterns = [
        r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)(?:from|with|where|for)',  # expression between keywords
        r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)$',  # expression at end of sentence
        r'([a-z0-9\^\(\)\+\-\*\/\.\s]+)\s*(?:from|with|between)\s+[-\d]',  # expression before range
        r'z\s*=\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # z = expression
        r'f\s*\(\s*x\s*(?:,\s*y)?\s*\)\s*=\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # f(x) = or f(x,y) = 
        r'y\s*=\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)'  # y = expression
    ]
    
    for pattern in complex_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            # Clean up the expression
            return re.sub(r'\s+', '', expr)  # Remove all whitespace from expression
    
    # Define original patterns (as fallback)
    expression_patterns = [
        r'sin\s*\(\s*x\s*\)',  # sin(x)
        r'cos\s*\(\s*x\s*\)',  # cos(x)
        r'tan\s*\(\s*x\s*\)',  # tan(x)
        r'x\s*\^\s*\d+',       # x^2, x^3, etc.
        r'e\s*\^\s*x',         # e^x
        r'log\s*\(\s*x\s*\)',  # log(x)
        r'ln\s*\(\s*x\s*\)',   # ln(x)
    ]
    
    # Try each pattern
    for pattern in expression_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    # Check for common expression keywords
    expression_keywords = [
        "plot", "graph", "function", "expression", "formula", "equation"
    ]
    
    # Look for expressions in the vicinity of these keywords
    for keyword in expression_keywords:
        keyword_pos = text.lower().find(keyword)
        if keyword_pos != -1:
            # Extract the text after the keyword
            after_keyword = text[keyword_pos + len(keyword):].strip()
            # Look for a potential expression
            expr_match = re.search(r'[a-z0-9\^\(\)\+\-\*\/\.]+', after_keyword)
            if expr_match:
                return expr_match.group(0).strip()
    
    # Fallback heuristics
    if "sin" in text.lower():
        return "sin(x)"
    elif "cos" in text.lower():
        return "cos(x)"
    elif "plot" in text.lower():
        return "x^2"  # Default fallback
    
    return None

# Enhance the extract_3d_expression method to use the concept library
def extract_3d_expression(self, text: str) -> str:
    """
    Extract a 3D function expression from text.
    
    Args:
        text: Text to extract expression from
        
    Returns:
        Extracted expression or None if no expression found
    """
    # Check for conditional expressions first
    conditional_expr = self.handle_conditional_expression(text)
    if conditional_expr:
        return conditional_expr
    
    # Look up mathematical concepts
    concept_formula = self.lookup_concept(text)
    if concept_formula:
        return concept_formula
        
    # Check for expressions enclosed in quotes first (most reliable)
    quoted_expr = re.search(r'["\']([^"\']+)["\']', text, re.IGNORECASE)
    if quoted_expr:
        return quoted_expr.group(1).strip()
    
    # Advanced patterns for 3D expressions
    advanced_patterns = [
        r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)(?:from|with|where|for)',  # expression between keywords
        r'(?:of|for|=|:)\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)$',  # expression at end of sentence
        r'3d\s+(?:plot|graph|visualization|surface)\s+(?:of|for)?\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # 3D plot of ...
        r'surface\s+(?:plot|graph)?\s+(?:of|for)?\s*([a-z0-9\^\(\)\+\-\*\/\.\s]+)',  # surface plot of ...
    ]
    
    for pattern in advanced_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            # Clean up the expression
            return re.sub(r'\s+', '', expr)  # Remove all whitespace from expression
    
    # Check for common 3D function pattern: z = f(x,y)
    z_pattern = r'z\s*=\s*([^\n]+)'
    match = re.search(z_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Check for specific 3D function patterns
    patterns = [
        r'sin\s*\(\s*x\s*\)\s*\*\s*cos\s*\(\s*y\s*\)',  # sin(x)*cos(y)
        r'x\s*\^\s*2\s*\+\s*y\s*\^\s*2',                # x^2 + y^2
        r'x\s*\^\s*\d+\s*\+\s*y\s*\^\s*\d+',            # x^n + y^m
        r'([a-z0-9\^\(\)\+\-\*\/\.]+)',                 # Generic expression pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    # Check for 3D visualization with sin and cos
    if "3d" in text.lower() and "sin" in text.lower() and "cos" in text.lower():
        return "sin(x)*cos(y)"
    
    # Default 3D function
    return "x^2 + y^2"

# Update the sanitize_expression method to handle numpy functions and special expressions
def sanitize_expression(self, expression: str) -> str:
    """
    Sanitize and standardize a mathematical expression.
    
    Args:
        expression: The raw expression string
        
    Returns:
        Sanitized expression
    """
    if not expression:
        return None
    
    # Check if it's already a numpy where expression
    if expression.startswith("np.where"):
        return expression
        
    # Replace caret notation with double asterisk for Python
    expression = re.sub(r'(\w+)\s*\^\s*(\d+)', r'\1**\2', expression)
    
    # Replace any remaining ^ with ** for Python
    expression = expression.replace('^', '**')
    
    # Handle special functions and constants
    expression = (expression
                 .replace('pi', 'np.pi')
                 .replace('sin(', 'np.sin(')
                 .replace('cos(', 'np.cos(')
                 .replace('tan(', 'np.tan(')
                 .replace('exp(', 'np.exp(')
                 .replace('log(', 'np.log(')
                 .replace('sqrt(', 'np.sqrt(')
                 .replace('abs(', 'np.abs('))
    
    # If the expression contains 'np.' already, don't modify it again
    if 'np.' not in expression:
        # Standardize the expression for evaluation
        # Add explicit multiplication where implied
        expression = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expression)
        expression = re.sub(r'(\))([a-zA-Z\(])', r'\1*\2', expression)
    
    # Remove all whitespace
    expression = re.sub(r'\s+', '', expression)
    
    return expression

# Add support for numpy at the start of the analyze_plot_request method
def analyze_plot_request(self, text: str) -> Dict[str, Any]:
    """
    Analyze text to identify plot requests and extract parameters.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Add numpy import instruction to ensure conditional expressions work
    numpy_import_instruction = "import numpy as np"
    
    # Try direct extraction of complex expressions first
    expression_indicators = [
        "x^", "sin(", "cos(", "log(", "exp(", "(x", "z=", "f(x", "y="
    ]
    
    # Look for mathematical concepts first
    concept_formula = self.lookup_concept(text)
    if concept_formula:
        logger.info(f"Found mathematical concept formula: {concept_formula}")
        sanitized_formula = self.sanitize_expression(concept_formula)
        return {
            "success": True,
            "is_visualization_request": True,
            "plot_type": "function_3d" if "y" in concept_formula else "function_2d",
            "parameters": {
                "expression": sanitized_formula,
                "numpy_import": numpy_import_instruction
            },
            "analysis": {
                "is_visualization_request": True,
                "plot_type": "function_3d" if "y" in concept_formula else "function_2d",
                "mathematical_expression": concept_formula,
                "concept_match": True
            }
        }
        
    # Check for conditional expressions
    conditional_expr = self.handle_conditional_expression(text)
    if conditional_expr:
        logger.info(f"Found conditional expression: {conditional_expr}")
        return {
            "success": True,
            "is_visualization_request": True,
            "plot_type": "function_3d" if "y" in text.lower() else "function_2d",
            "parameters": {
                "expression": conditional_expr,
                "numpy_import": numpy_import_instruction
            },
            "analysis": {
                "is_visualization_request": True,
                "plot_type": "function_3d" if "y" in text.lower() else "function_2d",
                "mathematical_expression": conditional_expr,
                "is_conditional": True
            }
        }
    
    # Direct extraction for 3D plots
    if "3d" in text.lower() and any(indicator in text.lower() for indicator in expression_indicators):
        logger.info("Attempting direct extraction for 3D plot")
        direct_extraction = True
        extracted_expr = self.extract_3d_expression(text)
        if extracted_expr:
            return {
                "success": True,
                "is_visualization_request": True,
                "plot_type": "function_3d",
                "parameters": {
                    "expression": self.sanitize_expression(extracted_expr),
                    "numpy_import": numpy_import_instruction
                },
                "analysis": {
                    "is_visualization_request": True,
                    "plot_type": "function_3d",
                    "mathematical_expression": extracted_expr
                }
            }
            
    # Direct extraction for 2D plots
    elif any(indicator in text.lower() for indicator in expression_indicators):
        logger.info("Attempting direct extraction for 2D plot")
        direct_extraction = True
        extracted_expr = self.extract_expression(text)
        if extracted_expr:
            return {
                "success": True,
                "is_visualization_request": True,
                "plot_type": "function_2d",
                "parameters": {
                    "expression": self.sanitize_expression(extracted_expr)
                },
                "analysis": {
                    "is_visualization_request": True,
                    "plot_type": "function_2d",
                    "mathematical_expression": extracted_expr
                }
            }
        
    # Special case for helix which has a specific structure
    if "helix" in text.lower() and all(term in text.lower() for term in ["x", "y", "z", "cos", "sin", "t"]):
        return {
            "success": True,
            "is_visualization_request": True,
            "plot_type": "parametric_3d",
            "parameters": {
                "x_expression": "cos(t)",
                "y_expression": "sin(t)",
                "z_expression": "t",
                "t_min": 0,
                "t_max": 6.28,
                "title": "3D Helix"
            },
            "analysis": {
                "is_visualization_request": True,
                "plot_type": "parametric_3d"
            }
        }
            
    # Continue with LLM analysis if direct extraction failed or wasn't attempted
    # Ensure we have the LLM agent
    if not self.llm_agent:
        try:
            from api.rest.server import get_core_llm_agent
            self.llm_agent = get_core_llm_agent()
            if not self.llm_agent:
                return {
                    "success": False,
                    "error": "LLM agent not available for text analysis"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Could not access LLM agent: {str(e)}"
            }
    
    try:
        # Use the LLM to analyze the request
        llm_response = self.llm_agent.generate_response(
            prompt=text,
            system_prompt=self.system_prompt,
            temperature=0.2
        )
        
        if not llm_response.get("success", False):
            return {
                "success": False,
                "error": f"LLM analysis failed: {llm_response.get('error', 'Unknown error')}"
            }
        
        response_text = llm_response.get("response", "")
        
        # Extract JSON from the response
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        
        if json_match:
            json_text = json_match.group(1).strip()
        else:
            # Try to find any JSON-like structure
            json_match = re.search(r'(\{[\s\S]*\})', response_text)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                # Just use the full response text and hope for the best
                json_text = response_text
        
        # Try to clean up the JSON text to make it more parseable
        # Remove any explanatory text before or after the JSON
        json_text = re.sub(r'^[^{]*', '', json_text)
        json_text = re.sub(r'[^}]*$', '', json_text)
        
        try:
            analysis = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response: {response_text}")
            
            # Manual fallback detection for common visualization keywords
            if any(keyword in text.lower() for keyword in ["plot", "graph", "visualize", "chart", "draw", "function"]):
                # Determine if this is 2D or 3D
                is_3d = "3d" in text.lower() or "3-d" in text.lower() or "three dimension" in text.lower()
                
                # Create a basic analysis structure
                analysis = {
                    "is_visualization_request": True,
                    "plot_type": "function_3d" if is_3d else "function_2d"
                }
                
                # Try direct extraction if this is likely a visualization
                extracted_expr = None
                if is_3d:
                    extracted_expr = self.extract_3d_expression(text)
                else:
                    extracted_expr = self.extract_expression(text)
                    
                if extracted_expr:
                    # Use sanitized version for parameters
                    analysis["parameters"] = {
                        "expression": self.sanitize_expression(extracted_expr)
                    }
                    # Save original expression for display/reference
                    analysis["mathematical_expression"] = extracted_expr
            else:
                # Return a simple structure with the raw text
                return {
                    "success": True,
                    "is_visualization_request": False,
                    "raw_llm_response": response_text
                }
        
        # Determine if this is a visualization request
        is_visualization = analysis.get("is_visualization_request", False)
        if isinstance(is_visualization, str):
            is_visualization = is_visualization.lower() == "yes"
        
        # Fallback detection if LLM didn't identify it correctly
        if not is_visualization:
            # Check for common visualization keywords
            keywords = ["plot", "graph", "visualize", "chart", "draw", "show me", "display"]
            if any(keyword in text.lower() for keyword in keywords):
                is_visualization = True
                analysis["is_visualization_request"] = True
                # Default to function_2d if no plot type
                if "plot_type" not in analysis:
                    analysis["plot_type"] = "function_2d"
                # Initialize parameters if not present
                if "parameters" not in analysis:
                    analysis["parameters"] = {}
        
        if not is_visualization:
            return {
                "success": True,
                "is_visualization_request": False,
                "plot_type": None,
                "parameters": {},
                "analysis": analysis
            }
        
        # Extract the plot type and parameters
        plot_type = analysis.get("plot_type", "function_2d")
        if isinstance(plot_type, str):
            plot_type = plot_type.strip()  # Remove leading/trailing spaces
        parameters = analysis.get("parameters", {})
        
        # Fallback extraction if parameters are incomplete
        if plot_type == "function_2d" and "expression" not in parameters:
            extracted_expr = self.extract_expression(text)
            if extracted_expr:
                parameters["expression"] = self.sanitize_expression(extracted_expr)
                # Save original expression for display/reference
                analysis["mathematical_expression"] = extracted_expr
        
        elif plot_type == "functions_2d":
            if "expressions" not in parameters:
                # Try to extract multiple expressions
                expressions = self.extract_multiple_expressions(text)
                if expressions:
                    # Sanitize all expressions
                    parameters["expressions"] = [self.sanitize_expression(expr) for expr in expressions]
                    # Save original expressions for display/reference
                    analysis["mathematical_expressions"] = expressions
                elif "expression" in parameters:
                    # Convert single expression to a list
                    parameters["expressions"] = [parameters["expression"]]
        
        elif plot_type == "function_3d" and "expression" not in parameters:
            extracted_expr = self.extract_3d_expression(text)
            if extracted_expr:
                parameters["expression"] = self.sanitize_expression(extracted_expr)
                # Save original expression for display/reference
                analysis["mathematical_expression"] = extracted_expr
        
        elif plot_type == "parametric_3d":
            if not all(k in parameters for k in ["x_expression", "y_expression", "z_expression"]):
                parametric_exprs = self.extract_parametric_3d(text)
                if parametric_exprs:
                    # Sanitize all expressions
                    for key, expr in parametric_exprs.items():
                        parametric_exprs[key] = self.sanitize_expression(expr)
                    parameters.update(parametric_exprs)
        
        # Ensure we have the minimal required parameters for each plot type
        if plot_type == "function_2d":
            if "expression" not in parameters:
                return {
                    "success": False,
                    "error": "Missing required parameter 'expression' for function_2d plot",
                    "is_visualization_request": True,
                    "plot_type": plot_type,
                    "parameters": parameters
                }
        
        elif plot_type == "functions_2d":
            if "expressions" not in parameters:
                # Try to extract multiple expressions
                if "expression" in parameters:
                    # Convert single expression to a list
                    parameters["expressions"] = [parameters["expression"]]
                else:
                    return {
                        "success": False,
                        "error": "Missing required parameter 'expressions' for functions_2d plot",
                        "is_visualization_request": True,
                        "plot_type": plot_type,
                        "parameters": parameters
                    }
        
        elif plot_type == "function_3d":
            if "expression" not in parameters:
                return {
                    "success": False,
                    "error": "Missing required parameter 'expression' for function_3d plot",
                    "is_visualization_request": True,
                    "plot_type": plot_type,
                    "parameters": parameters
                }
        
        elif plot_type == "parametric_3d":
            if not all(k in parameters for k in ["x_expression", "y_expression", "z_expression"]):
                return {
                    "success": False,
                    "error": "Missing required parameters 'x_expression', 'y_expression', 'z_expression' for parametric_3d plot",
                    "is_visualization_request": True,
                    "plot_type": plot_type,
                    "parameters": parameters
                }
        
        # Format the return object
        return {
            "success": True,
            "is_visualization_request": True,
            "plot_type": plot_type,
            "parameters": parameters,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error in plot request analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error analyzing plot request: {str(e)}"
        } 
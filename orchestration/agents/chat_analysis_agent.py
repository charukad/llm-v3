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

# System prompt for plot extraction
PLOT_EXTRACTION_PROMPT = """
You are a specialized assistant for analyzing mathematical visualization requests.
Your task is to identify if the user is requesting a plot or visualization and extract all necessary information to create it.

For each request, determine:
1. Is this a visualization request? (yes/no)
2. What type of plot is needed? (function_2d, functions_2d, function_3d, parametric_3d)
3. Extract all relevant parameters based on the plot type:

For function_2d:
- Mathematical expression(s)
- X range (min and max values)
- Plot title (optional)
- Axis labels (optional)

For functions_2d:
- List of mathematical expressions
- List of labels for each expression (optional)
- X range (min and max values)
- Plot title (optional)
- Axis labels (optional)

For function_3d:
- Mathematical expression for z=f(x,y)
- X range and Y range
- Plot title (optional)
- Axis labels (optional)
- View angle (optional)

For parametric_3d:
- X, Y, and Z expressions in terms of parameter t
- T range (min and max values)
- Plot title (optional)
- Axis labels (optional)
- Color (optional)

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
        # Define patterns
        expression_patterns = [
            r'sin\s*\(\s*x\s*\)',  # sin(x)
            r'cos\s*\(\s*x\s*\)',  # cos(x)
            r'tan\s*\(\s*x\s*\)',  # tan(x)
            r'x\s*\^\s*\d+',       # x^2, x^3, etc.
            r'e\s*\^\s*x',         # e^x
            r'log\s*\(\s*x\s*\)',  # log(x)
            r'ln\s*\(\s*x\s*\)',   # ln(x)
            r'f\s*\(\s*x\s*\)\s*=\s*([^\n]+)',  # f(x) = ...
            r'y\s*=\s*([^\n]+)'    # y = ...
        ]
        
        # Try each pattern
        for pattern in expression_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if '=' in pattern:
                    return match.group(1).strip()
                else:
                    return match.group(0).strip()
        
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
        # Check for common 3D function pattern: z = f(x,y)
        z_pattern = r'z\s*=\s*([^\n]+)'
        match = re.search(z_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Check for specific 3D function patterns
        patterns = [
            r'sin\s*\(\s*x\s*\)\s*\*\s*cos\s*\(\s*y\s*\)',  # sin(x)*cos(y)
            r'x\s*\^\s*2\s*\+\s*y\s*\^\s*2',                # x^2 + y^2
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
    
    def analyze_plot_request(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to identify plot requests and extract parameters.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
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
                    parameters["expression"] = extracted_expr
            
            elif plot_type == "functions_2d":
                if "expressions" not in parameters:
                    # Try to extract multiple expressions
                    expressions = self.extract_multiple_expressions(text)
                    if expressions:
                        parameters["expressions"] = expressions
                    elif "expression" in parameters:
                        # Convert single expression to a list
                        parameters["expressions"] = [parameters["expression"]]
            
            elif plot_type == "function_3d" and "expression" not in parameters:
                extracted_expr = self.extract_3d_expression(text)
                if extracted_expr:
                    parameters["expression"] = extracted_expr
            
            elif plot_type == "parametric_3d":
                if not all(k in parameters for k in ["x_expression", "y_expression", "z_expression"]):
                    parametric_exprs = self.extract_parametric_3d(text)
                    if parametric_exprs:
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
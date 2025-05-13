"""
Mathematical Computation Agent - processes mathematical queries.

This agent handles the symbolic computation of mathematical expressions,
providing solutions, step-by-step explanations, and verification.
"""

import sympy as sp
from typing import Dict, List, Optional, Union, Any
import logging
import uuid
import json

from math_processing.expressions.latex_parser import parse_math_expression
from math_processing.expressions.normalizer import normalize_expression
from math_processing.expressions.converters import convert_expression
from math_processing.expressions.comparators import compare_expressions
from math_processing.computation.sympy_wrapper import SymbolicProcessor


class MathComputationAgent:
    """Mathematical Computation Agent for symbolic mathematics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Math Computation Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.id = f"math_computation_agent_{uuid.uuid4().hex[:8]}"
        self.symbolic_processor = SymbolicProcessor()
    
    def process_expression(self, expression: str, operation: str, 
                         parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a mathematical expression with the given operation.
        
        Args:
            expression: Mathematical expression (typically in LaTeX format)
            operation: Operation to perform (e.g., "solve", "differentiate")
            parameters: Optional parameters for the operation
            
        Returns:
            Processing result
        """
        # Create a computation request
        request = {
            "operation": operation,
            "expression": expression,
            "format": "latex",
            "parameters": parameters or {},
            "output_format": "latex"
        }
        
        # Process the request
        return self._handle_compute_request(request)
    
    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming messages from the message bus.
        
        Args:
            message: Message in the standardized MCP format
            
        Returns:
            Response message
        """
        # Extract message content
        header = message.get("header", {})
        body = message.get("body", {})
        
        # Log incoming message
        self.logger.info(f"Received message: {header.get('message_type')}")
        
        # Process based on message type
        message_type = header.get("message_type", "")
        
        if message_type == "compute_request":
            response_body = self._handle_compute_request(body)
        elif message_type == "parse_request":
            response_body = self._handle_parse_request(body)
        elif message_type == "verify_request":
            response_body = self._handle_verify_request(body)
        else:
            response_body = {
                "success": False,
                "error": f"Unsupported message type: {message_type}"
            }
        
        # Create response message
        response = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "correlation_id": header.get("message_id"),
                "sender": self.id,
                "recipient": header.get("sender"),
                "message_type": f"{message_type}_response",
                "timestamp": self._get_timestamp()
            },
            "body": response_body
        }
        
        return response
    
    def _handle_compute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a computation request.
        
        Args:
            request: Computation request body
            
        Returns:
            Computation response body
        """
        try:
            # Extract parameters
            operation = request.get("operation", "")
            expression_str = request.get("expression", "")
            parameters = request.get("parameters", {})
            
            # Parse the expression if in LaTeX format
            if request.get("format", "latex") == "latex":
                parse_result = parse_math_expression(expression_str)
                if not parse_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to parse expression: {parse_result['error']}"
                    }
                expression = parse_result["expression"]
            else:
                # Assume expression is already a SymPy expression (from another service)
                expression = expression_str
            
            # Normalize the expression
            normalize_result = normalize_expression(expression)
            if not normalize_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to normalize expression: {normalize_result['error']}"
                }
            expression = normalize_result["expression"]
            
            # Perform the requested operation
            if operation == "solve":
                result = self.symbolic_processor.solve_equation(
                    expression, 
                    parameters.get("variable")
                )
            elif operation == "differentiate":
                result = self.symbolic_processor.differentiate(
                    expression, 
                    parameters.get("variable"), 
                    parameters.get("order", 1)
                )
            elif operation == "integrate":
                result = self.symbolic_processor.integrate(
                    expression, 
                    parameters.get("variable"), 
                    parameters.get("lower_bound"), 
                    parameters.get("upper_bound")
                )
            elif operation == "evaluate":
                result = self.symbolic_processor.evaluate(
                    expression, 
                    parameters.get("values", {})
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}"
                }
            
            # Convert result to requested format
            output_format = request.get("output_format", "latex")
            conversion_result = convert_expression(result, output_format)
            
            if not conversion_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to convert result: {conversion_result['error']}"
                }
            
            # Return successful result
            return {
                "success": True,
                "operation": operation,
                "result": conversion_result["result"],
                "result_latex": self.symbolic_processor.to_latex(result),
                "steps": self._get_computation_steps(operation, expression, result, parameters),
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in compute request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _handle_parse_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a parsing request.
        
        Args:
            request: Parsing request body
            
        Returns:
            Parsing response body
        """
        try:
            # Extract parameters
            expression_str = request.get("expression", "")
            
            # Parse the expression
            parse_result = parse_math_expression(expression_str)
            
            if not parse_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to parse expression: {parse_result['error']}"
                }
            
            # Extract variables from the parsed expression
            variables = parse_result["variables"]
            
            # Return successful result
            return {
                "success": True,
                "parsed_expression": self.symbolic_processor.to_latex(parse_result["expression"]),
                "variables": list(variables.keys()),
                "domain": self._detect_domain(parse_result["expression"]),
                "complexity": self._estimate_complexity(parse_result["expression"]),
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in parse request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _handle_verify_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a verification request to check if two expressions are equivalent.
        
        Args:
            request: Verification request body
            
        Returns:
            Verification response body
        """
        try:
            # Extract parameters
            expr1_str = request.get("expression1", "")
            expr2_str = request.get("expression2", "")
            method = request.get("method", "symbolic")
            
            # Parse the expressions
            parse_result1 = parse_math_expression(expr1_str)
            if not parse_result1["success"]:
                return {
                    "success": False,
                    "error": f"Failed to parse first expression: {parse_result1['error']}"
                }
            
            parse_result2 = parse_math_expression(expr2_str)
            if not parse_result2["success"]:
                return {
                    "success": False,
                    "error": f"Failed to parse second expression: {parse_result2['error']}"
                }
            
            # Compare the expressions
            comparison_result = compare_expressions(
                parse_result1["expression"],
                parse_result2["expression"],
                method
            )
            
            # Return the comparison result
            return {
                "success": True,
                "equivalent": comparison_result["equivalent"],
                "method": comparison_result["method"],
                "explanation": comparison_result["explanation"],
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in verify request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_computation_steps(self, 
                              operation: str, 
                              expression: Union[sp.Expr, sp.Eq], 
                              result: Any, 
                              parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate step-by-step explanation of the computation.
        
        Args:
            operation: Type of operation performed
            expression: Original expression
            result: Computation result
            parameters: Additional parameters for the operation
            
        Returns:
            List of computation steps, each with an expression and explanation
        """
        # This is a simplified implementation
        # A full implementation would have domain-specific step generation
        
        steps = []
        
        # Add initial step
        steps.append({
            "step_number": 1,
            "expression_latex": self.symbolic_processor.to_latex(expression),
            "explanation": f"Start with the given expression"
        })
        
        # Operation-specific steps would be added here in a full implementation
        # This is just a placeholder
        if operation == "solve":
            steps.append({
                "step_number": 2,
                "expression_latex": self.symbolic_processor.to_latex(result),
                "explanation": f"Solution found"
            })
        elif operation == "differentiate":
            steps.append({
                "step_number": 2,
                "expression_latex": self.symbolic_processor.to_latex(result),
                "explanation": f"Apply differentiation"
            })
        elif operation == "integrate":
            steps.append({
                "step_number": 2,
                "expression_latex": self.symbolic_processor.to_latex(result),
                "explanation": f"Apply integration"
            })
        
        return steps
    
    def _detect_domain(self, expression: Union[sp.Expr, sp.Eq]) -> str:
        """
        Detect the mathematical domain of an expression.
        
        Args:
            expression: SymPy expression or equation
            
        Returns:
            Detected domain as a string
        """
        # This is a simplified implementation
        # A full implementation would use more sophisticated detection
        
        # Check for calculus operations (derivatives, integrals)
        if expression.has(sp.Derivative) or expression.has(sp.Integral):
            return "calculus"
        
        # Check for matrices
        if expression.has(sp.Matrix):
            return "linear_algebra"
        
        # Check for statistics functions
        stats_funcs = [sp.stats.Normal, sp.stats.Uniform, sp.stats.Poisson]
        if any(expression.has(func) for func in stats_funcs):
            return "statistics"
        
        # Default to algebra
        return "algebra"
    
    def _estimate_complexity(self, expression: Union[sp.Expr, sp.Eq]) -> int:
        """
        Estimate the complexity of an expression based on its structure.
        
        Args:
            expression: SymPy expression or equation
            
        Returns:
            Complexity score (higher is more complex)
        """
        # This is a simplified implementation
        # A full implementation would have more sophisticated metrics
        
        # Count the number of operations as a simple complexity metric
        if isinstance(expression, sp.Eq):
            # For equations, count operations on both sides
            count = len(expression.lhs.args) + len(expression.rhs.args)
        else:
            # For expressions, count operations
            count = len(expression.args)
        
        # Add complexity for specific operations
        if expression.has(sp.Derivative):
            count += 5
        if expression.has(sp.Integral):
            count += 5
        if expression.has(sp.Matrix):
            count += 5
        
        return count
    
    def _get_timestamp(self) -> str:
        """
        Get the current timestamp in ISO format.
        
        Returns:
            Current timestamp as string
        """
        from datetime import datetime
        return datetime.now().isoformat()

# Backwards compatibility alias for older imports
MathAgent = MathComputationAgent

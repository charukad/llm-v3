"""
Mathematical Knowledge Agent - enhanced Math Agent with knowledge base integration.

This module extends the Math Computation Agent with access to the mathematical
knowledge base, enabling context-aware mathematical problem solving with
reference to concepts, theorems, and formulas.
"""

import sympy as sp
from typing import Dict, List, Optional, Union, Any
import logging
import uuid
import json

from math_processing.agent.math_agent import MathAgent
from math_processing.knowledge.knowledge_base import MathKnowledgeBase
from math_processing.classification.domain_classifier import MathDomainClassifier


class MathKnowledgeAgent(MathAgent):
    """Enhanced Math Agent with knowledge base integration."""
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None,
                knowledge_base_path: Optional[str] = None):
        """
        Initialize the Math Knowledge Agent.
        
        Args:
            config: Optional configuration dictionary
            knowledge_base_path: Path to the knowledge base file
        """
        # Initialize the base MathAgent
        super().__init__(config)
        
        # Override the ID for this specialized agent
        self.id = f"math_knowledge_agent_{uuid.uuid4().hex[:8]}"
        
        # Initialize the knowledge base
        self.knowledge_base = MathKnowledgeBase(knowledge_base_path)
        
        # Initialize the domain classifier
        self.domain_classifier = MathDomainClassifier()
        
        self.logger.info("Math Knowledge Agent initialized")
    
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
        
        if message_type == "knowledge_query":
            response_body = self._handle_knowledge_query(body)
        elif message_type == "concept_lookup":
            response_body = self._handle_concept_lookup(body)
        elif message_type == "formula_lookup":
            response_body = self._handle_formula_lookup(body)
        elif message_type == "domain_classify":
            response_body = self._handle_domain_classification(body)
        else:
            # For other message types, use the base MathAgent handler
            return super().handle_message(message)
        
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
    
    def _handle_knowledge_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a knowledge query.
        
        Args:
            request: Knowledge query request body
            
        Returns:
            Knowledge query response body
        """
        try:
            # Extract query and domain
            query = request.get("query", "")
            domain = request.get("domain", None)
            max_results = request.get("max_results", 5)
            
            # If domain is not provided, try to classify the query
            if domain is None:
                classification = self.domain_classifier.classify_query(query)
                domain = classification["primary_domain"]
            
            # Search for relevant concepts
            concepts = self.knowledge_base.search_concepts(query, domain, max_results)
            
            # Search for relevant theorems
            theorems = self.knowledge_base.search_theorems(query, domain, max_results)
            
            # Search for relevant formulas
            formulas = self.knowledge_base.search_formulas(query, domain, max_results)
            
            # Return the results
            return {
                "success": True,
                "query": query,
                "domain": domain,
                "concepts": concepts,
                "theorems": theorems,
                "formulas": formulas,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in knowledge query: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _handle_concept_lookup(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a concept lookup.
        
        Args:
            request: Concept lookup request body
            
        Returns:
            Concept lookup response body
        """
        try:
            # Extract concept name or ID
            concept_id = request.get("concept_id")
            concept_name = request.get("concept_name")
            
            # Look up the concept
            concept = None
            if concept_id:
                concept = self.knowledge_base.get_concept_by_id(concept_id)
            elif concept_name:
                concept = self.knowledge_base.get_concept_by_name(concept_name)
            
            if concept:
                # Get related concepts
                related_concepts = []
                if "id" in concept:
                    related_concepts = self.knowledge_base.get_related_concepts(concept["id"])
                
                return {
                    "success": True,
                    "concept": concept,
                    "related_concepts": related_concepts,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "error": f"Concept not found: {concept_id or concept_name}"
                }
        except Exception as e:
            self.logger.error(f"Error in concept lookup: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _handle_formula_lookup(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a formula lookup.
        
        Args:
            request: Formula lookup request body
            
        Returns:
            Formula lookup response body
        """
        try:
            # Extract formula name or ID
            formula_id = request.get("formula_id")
            formula_name = request.get("formula_name")
            
            # Look up the formula
            formula = None
            if formula_id:
                formula = self.knowledge_base.get_formula_by_id(formula_id)
            elif formula_name:
                formula = self.knowledge_base.get_formula_by_name(formula_name)
            
            if formula:
                return {
                    "success": True,
                    "formula": formula,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "error": f"Formula not found: {formula_id or formula_name}"
                }
        except Exception as e:
            self.logger.error(f"Error in formula lookup: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _handle_domain_classification(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle domain classification.
        
        Args:
            request: Domain classification request body
            
        Returns:
            Domain classification response body
        """
        try:
            # Extract query
            query = request.get("query", "")
            include_details = request.get("include_details", False)
            
            # Classify the query
            classification = self.domain_classifier.classify_query(query, include_details)
            
            # Get domain description and examples
            domain = classification["primary_domain"]
            description = self.domain_classifier.get_domain_description(domain)
            examples = self.domain_classifier.get_domain_examples(domain)
            
            # Add to the classification results
            classification["domain_description"] = description
            classification["domain_examples"] = examples
            
            return {
                "success": True,
                "classification": classification,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error in domain classification: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def solve_with_context(self, 
                         query: str, 
                         expression: Union[str, sp.Expr, sp.Eq],
                         operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a mathematical problem with context from the knowledge base.
        
        Args:
            query: User query
            expression: Mathematical expression
            operation: Specific operation to perform (optional)
            
        Returns:
            Solution with context
        """
        try:
            # Classify the domain if operation is not specified
            if operation is None:
                classification = self.domain_classifier.classify_query(query)
                domain = classification["primary_domain"]
                
                # Infer operation based on query and domain
                operation = self._infer_operation(query, domain)
            else:
                # Use the provided operation but still classify for domain
                classification = self.domain_classifier.classify_query(query)
                domain = classification["primary_domain"]
            
            # Find relevant knowledge
            knowledge_query = {
                "query": query,
                "domain": domain,
                "max_results": 2  # Limit to the most relevant results
            }
            knowledge = self._handle_knowledge_query(knowledge_query)
            
            # Parse the expression if it's a string
            if isinstance(expression, str):
                parse_result = self._parse_expression(expression)
                if not parse_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to parse expression: {parse_result['error']}"
                    }
                expr = parse_result["expression"]
            else:
                expr = expression
            
            # Solve the problem based on the operation
            result = self._solve_problem(expr, operation, domain)
            
            # Enhance the result with knowledge context
            result["domain"] = domain
            result["operation"] = operation
            result["knowledge_context"] = {
                "concepts": knowledge.get("concepts", []),
                "theorems": knowledge.get("theorems", []),
                "formulas": knowledge.get("formulas", [])
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error in solve_with_context: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _infer_operation(self, query: str, domain: str) -> str:
        """
        Infer the mathematical operation from the query and domain.
        
        Args:
            query: User query
            domain: Mathematical domain
            
        Returns:
            Inferred operation
        """
        query_lower = query.lower()
        
        # Check for operation keywords in the query
        if any(keyword in query_lower for keyword in ["solve", "find", "solution", "root"]):
            if domain == "algebra":
                return "solve"
        
        if any(keyword in query_lower for keyword in ["derivative", "differentiate", "slope"]):
            return "differentiate"
        
        if any(keyword in query_lower for keyword in ["integral", "integrate", "antiderivative"]):
            return "integrate"
        
        if any(keyword in query_lower for keyword in ["limit", "approaches", "tends to"]):
            return "limit"
        
        if any(keyword in query_lower for keyword in ["factor", "factorize"]):
            return "factor"
        
        if any(keyword in query_lower for keyword in ["expand", "distribute"]):
            return "expand"
        
        if any(keyword in query_lower for keyword in ["simplify", "simplification"]):
            return "simplify"
        
        # Default operations based on domain
        domain_defaults = {
            "algebra": "solve",
            "calculus": "differentiate",
            "linear_algebra": "determinant",
            "statistics": "analyze",
            "geometry": "calculate",
            "number_theory": "analyze",
            "trigonometry": "evaluate",
            "discrete_math": "analyze"
        }
        
        return domain_defaults.get(domain, "analyze")
    
    def _parse_expression(self, expression_str: str) -> Dict[str, Any]:
        """
        Parse a mathematical expression string.
        
        Args:
            expression_str: Expression string
            
        Returns:
            Parsing result
        """
        try:
            # Check if the string contains an equation
            if "=" in expression_str:
                lhs, rhs = expression_str.split("=", 1)
                expr = sp.Eq(sp.sympify(lhs.strip()), sp.sympify(rhs.strip()))
            else:
                expr = sp.sympify(expression_str)
            
            return {
                "success": True,
                "expression": expr,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "expression": None,
                "error": str(e)
            }
    
    def _solve_problem(self, 
                     expression: Union[sp.Expr, sp.Eq], 
                     operation: str,
                     domain: str) -> Dict[str, Any]:
        """
        Solve a mathematical problem.
        
        Args:
            expression: Mathematical expression
            operation: Operation to perform
            domain: Mathematical domain
            
        Returns:
            Solution result
        """
        try:
            # Extract variable (for operations that need it)
            variable = None
            if isinstance(expression, sp.Eq):
                all_symbols = expression.lhs.free_symbols.union(expression.rhs.free_symbols)
                if all_symbols:
                    variable = list(all_symbols)[0]
            elif expression.free_symbols:
                variable = list(expression.free_symbols)[0]
            
            # Perform the operation
            if operation == "solve":
                result = self.symbolic_processor.solve_equation(expression, variable)
                return {
                    "success": True,
                    "solutions": result,
                    "variable": variable,
                    "error": None
                }
            
            elif operation == "differentiate":
                derivative = self.symbolic_processor.differentiate(expression, variable)
                return {
                    "success": True,
                    "derivative": derivative,
                    "variable": variable,
                    "error": None
                }
            
            elif operation == "integrate":
                integral = self.symbolic_processor.integrate(expression, variable)
                return {
                    "success": True,
                    "integral": integral,
                    "variable": variable,
                    "error": None
                }
            
            elif operation == "factor":
                factored = self.symbolic_processor.factor(expression)
                return {
                    "success": True,
                    "factored": factored,
                    "error": None
                }
            
            elif operation == "expand":
                expanded = self.symbolic_processor.expand(expression)
                return {
                    "success": True,
                    "expanded": expanded,
                    "error": None
                }
            
            elif operation == "simplify":
                simplified = self.symbolic_processor.simplify(expression)
                return {
                    "success": True,
                    "simplified": simplified,
                    "error": None
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}"
                }
                
        except Exception as e:
            self.logger.error(f"Error solving problem: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

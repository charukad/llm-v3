"""
Test for the Math Knowledge Agent.

This module tests the Math Knowledge Agent, which integrates mathematical
problem solving with the knowledge base.
"""

import pytest
import sympy as sp
from math_processing.agent.math_knowledge_agent import MathKnowledgeAgent
from math_processing.knowledge.knowledge_base import MathKnowledgeBase
import logging
import sys
import os
import tempfile
import json

# Configure logging for the test
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TestMathKnowledgeAgent:
    """Tests for the Math Knowledge Agent."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a temporary knowledge base file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        
        # Initialize with some test data
        test_data = {
            'concepts': {},
            'theorems': {},
            'formulas': {}
        }
        
        json.dump(test_data, self.temp_file)
        self.temp_file.close()
        
        # Initialize the knowledge base and agent
        self.math_knowledge_agent = MathKnowledgeAgent(knowledge_base_path=self.temp_file.name)
        self.knowledge_base = MathKnowledgeBase(self.temp_file.name)
        
        # Add some test concepts, theorems, and formulas
        self._add_test_knowledge()
    
    def teardown_method(self):
        """Clean up test environment."""
        # Delete the temporary file
        os.unlink(self.temp_file.name)
    
    def _add_test_knowledge(self):
        """Add test data to the knowledge base."""
        # Add a quadratic formula
        self.knowledge_base.add_formula({
            "name": "quadratic_formula",
            "display_name": "Quadratic Formula",
            "latex": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
            "description": "Formula for finding the roots of a quadratic equation ax^2 + bx + c = 0.",
            "domain": "algebra"
        })
        
        # Add a derivative rule
        self.knowledge_base.add_formula({
            "name": "power_rule",
            "display_name": "Power Rule (Differentiation)",
            "latex": "\\frac{d}{dx}[x^n] = nx^{n-1}",
            "description": "Rule for differentiating powers of a variable.",
            "domain": "calculus"
        })
        
        # Add a concept
        self.knowledge_base.add_concept({
            "name": "Quadratic Equation",
            "definition": "A quadratic equation is an equation of the form ax^2 + bx + c = 0, where a, b, and c are constants and a ≠ 0.",
            "examples": ["x^2 - 5x + 6 = 0", "2x^2 + 4x - 3 = 0"],
            "domain": "algebra"
        })
        
        # Add a theorem
        self.knowledge_base.add_theorem({
            "name": "Fundamental Theorem of Calculus",
            "statement": "If f is continuous on the closed interval [a, b] and F is an antiderivative of f on [a, b], then ∫[a,b] f(x) dx = F(b) - F(a).",
            "implications": "This theorem connects differentiation and integration as inverse operations.",
            "domain": "calculus"
        })
    
    def test_domain_classification(self):
        """Test domain classification."""
        # Create a message for domain classification
        message = {
            "header": {
                "message_id": "test_id",
                "sender": "test_sender",
                "message_type": "domain_classify"
            },
            "body": {
                "query": "Solve the quadratic equation x^2 - 5x + 6 = 0",
                "include_details": True
            }
        }
        
        # Handle the message
        response = self.math_knowledge_agent.handle_message(message)
        
        # Verify the response
        assert response["body"]["success"] is True, "Domain classification failed"
        assert response["body"]["classification"]["primary_domain"] == "algebra", "Failed to classify as algebra problem"
        assert "domain_description" in response["body"]["classification"], "No domain description included"
        assert "domain_examples" in response["body"]["classification"], "No domain examples included"
        
        # Log the results
        logging.info("Domain classification test successful")
        logging.info(f"Primary domain: {response['body']['classification']['primary_domain']}")
        logging.info(f"Domain description: {response['body']['classification']['domain_description']}")
    
    def test_formula_lookup(self):
        """Test formula lookup."""
        # Create a message for formula lookup
        message = {
            "header": {
                "message_id": "test_id",
                "sender": "test_sender",
                "message_type": "formula_lookup"
            },
            "body": {
                "formula_name": "quadratic_formula"
            }
        }
        
        # Handle the message
        response = self.math_knowledge_agent.handle_message(message)
        
        # Verify the response
        assert response["body"]["success"] is True, "Formula lookup failed"
        assert response["body"]["formula"]["name"] == "quadratic_formula", "Failed to find quadratic formula"
        assert "latex" in response["body"]["formula"], "No LaTeX representation in the formula"
        
        # Log the results
        logging.info("Formula lookup test successful")
        logging.info(f"Formula: {response['body']['formula']['display_name']}")
        logging.info(f"LaTeX: {response['body']['formula']['latex']}")
    
    def test_knowledge_query(self):
        """Test knowledge query."""
        # Create a message for knowledge query
        message = {
            "header": {
                "message_id": "test_id",
                "sender": "test_sender",
                "message_type": "knowledge_query"
            },
            "body": {
                "query": "quadratic equation",
                "domain": "algebra"
            }
        }
        
        # Handle the message
        response = self.math_knowledge_agent.handle_message(message)
        
        # Verify the response
        assert response["body"]["success"] is True, "Knowledge query failed"
        assert len(response["body"]["concepts"]) > 0, "No concepts returned"
        assert len(response["body"]["formulas"]) > 0, "No formulas returned"
        
        # Check that the quadratic formula was returned
        formula_names = [formula["name"] for formula in response["body"]["formulas"]]
        assert "quadratic_formula" in formula_names, "Quadratic formula not found in results"
        
        # Log the results
        logging.info("Knowledge query test successful")
        logging.info(f"Concepts found: {len(response['body']['concepts'])}")
        logging.info(f"Formulas found: {len(response['body']['formulas'])}")
        logging.info(f"Theorems found: {len(response['body']['theorems'])}")
    
    def test_solve_with_context(self):
        """Test solving with knowledge context."""
        # Define a problem to solve
        query = "Solve the quadratic equation x^2 - 5x + 6 = 0"
        expression = "x^2 - 5x + 6 = 0"
        
        # Solve with context
        result = self.math_knowledge_agent.solve_with_context(query, expression)
        
        # Verify the result
        assert result["success"] is True, f"Solving with context failed: {result.get('error')}"
        assert "solutions" in result, "No solutions in the result"
        assert "knowledge_context" in result, "No knowledge context in the result"
        
        # Check that the solutions are correct
        solutions = result["solutions"]
        assert len(solutions) == 2, f"Expected 2 solutions, got {len(solutions)}"
        assert 2 in solutions, "Solution x = 2 not found"
        assert 3 in solutions, "Solution x = 3 not found"
        
        # Check that knowledge context includes the quadratic formula
        formulas = result["knowledge_context"]["formulas"]
        formula_names = [formula["name"] for formula in formulas]
        assert "quadratic_formula" in formula_names, "Quadratic formula not in knowledge context"
        
        # Log the results
        logging.info("Solve with context test successful")
        logging.info(f"Solutions: {solutions}")
        logging.info(f"Formulas in context: {formula_names}")
    
    def test_compute_request_handling(self):
        """Test handling of compute requests."""
        # Create a compute request message
        message = {
            "header": {
                "message_id": "test_id",
                "sender": "test_sender",
                "message_type": "compute_request"
            },
            "body": {
                "operation": "differentiate",
                "expression": "x^3 + 2*x^2",
                "parameters": {
                    "variable": "x"
                },
                "format": "latex",
                "output_format": "latex"
            }
        }
        
        # Handle the message
        response = self.math_knowledge_agent.handle_message(message)
        
        # Verify the response
        assert response["body"]["success"] is True, f"Compute request failed: {response['body'].get('error')}"
        assert "result" in response["body"], "No result in the response"
        assert "steps" in response["body"], "No steps in the response"
        
        # Check that the derivative is correct
        # The result is in LaTeX format, but we can check if it contains the expected terms
        result_latex = response["body"]["result"]
        assert "3 x^{2}" in result_latex, "Term 3x^2 not found in derivative"
        assert "4 x" in result_latex, "Term 4x not found in derivative"
        
        # Log the results
        logging.info("Compute request test successful")
        logging.info(f"Result LaTeX: {result_latex}")
        for i, step in enumerate(response["body"]["steps"]):
            logging.info(f"Step {i+1}: {step['explanation']}")

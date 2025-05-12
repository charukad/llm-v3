"""
Integration Tests for Mathematical Components

This module provides comprehensive integration tests for the mathematical
components of the system, including the knowledge base, step-by-step solution
generator, and verification mechanisms.
"""

import unittest
import sympy as sp
from math_processing.knowledge.knowledge_base import MathKnowledgeBase
from math_processing.solutions.step_generator import SolutionGenerator
from math_processing.solutions.verifier import SolutionVerifier
from math_processing.agent.math_knowledge_agent import MathKnowledgeAgent

class MathematicalComponentsIntegrationTest(unittest.TestCase):
    """Integration tests for mathematical components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.knowledge_base = MathKnowledgeBase()
        self.solution_generator = SolutionGenerator(self.knowledge_base)
        self.solution_verifier = SolutionVerifier()
        self.math_agent = MathKnowledgeAgent()
    
    def test_quadratic_equation_solution(self):
        """Test the complete workflow for solving a quadratic equation."""
        # Define a quadratic equation problem
        expression = "2*x^2 - 5*x + 2 = 0"
        operation = "solve"
        variable = sp.Symbol('x')
        
        # Generate step-by-step solution
        steps = self.solution_generator.generate_solution_steps(
            expression, operation, [variable], {"domain": "algebra"}
        )
        
        # Verify that steps were generated
        self.assertGreater(len(steps), 0, "No solution steps were generated")
        
        # Extract the solution from the steps
        final_step = steps[-1]
        self.assertIn("output", final_step.to_dict(), "Final step missing output")
        
        # Convert the solution to a format for verification
        problem = {
            "problem_type": "quadratic_equation",
            "equation": expression,
            "variable": "x"
        }
        
        # Extract solutions from the final step
        # This is simplified; in practice we'd need more robust parsing
        solutions = []
        if "x = " in str(final_step.output_expr):
            for sol_str in str(final_step.output_expr).replace("x = ", "").split(" or "):
                solutions.append(sol_str.strip())
        
        solution = {
            "solutions": solutions
        }
        
        # Verify the solution
        verification = self.solution_verifier.verify_solution(
            problem, solution, "algebra", ["substitution"]
        )
        
        # Check that verification passed
        self.assertTrue(verification.is_correct, 
                       f"Solution verification failed: {verification.error_message}")
    
    def test_derivative_calculation(self):
        """Test the complete workflow for calculating a derivative."""
        # Define a derivative problem
        expression = "x^2 * sin(x)"
        operation = "differentiate"
        variable = sp.Symbol('x')
        
        # Generate step-by-step solution
        steps = self.solution_generator.generate_solution_steps(
            expression, operation, [variable], {"domain": "calculus"}
        )
        
        # Verify that steps were generated
        self.assertGreater(len(steps), 0, "No solution steps were generated")
        
        # Extract the solution from the steps
        final_step = steps[-1]
        self.assertIn("output", final_step.to_dict(), "Final step missing output")
        
        # Convert the solution to a format for verification
        problem = {
            "problem_type": "derivative",
            "function": expression,
            "variable": "x"
        }
        
        solution = {
            "derivative": str(final_step.output_expr)
        }
        
        # Verify the solution
        verification = self.solution_verifier.verify_solution(
            problem, solution, "calculus", ["derivative"]
        )
        
        # Check that verification passed
        self.assertTrue(verification.is_correct, 
                       f"Derivative verification failed: {verification.error_message}")
    
    def test_integration_calculation(self):
        """Test the complete workflow for calculating an integral."""
        # Define an integration problem
        expression = "x^2"
        operation = "integrate"
        variable = sp.Symbol('x')
        
        # Generate step-by-step solution
        steps = self.solution_generator.generate_solution_steps(
            expression, operation, [variable], {"domain": "calculus"}
        )
        
        # Verify that steps were generated
        self.assertGreater(len(steps), 0, "No solution steps were generated")
        
        # Extract the solution from the steps
        final_step = steps[-1]
        self.assertIn("output", final_step.to_dict(), "Final step missing output")
        
        # Convert the solution to a format for verification
        problem = {
            "problem_type": "integral",
            "function": expression,
            "variable": "x"
        }
        
        solution = {
            "integral": str(final_step.output_expr)
        }
        
        # Verify the solution
        verification = self.solution_verifier.verify_solution(
            problem, solution, "calculus", ["integral"]
        )
        
        # Check that verification passed
        self.assertTrue(verification.is_correct, 
                       f"Integration verification failed: {verification.error_message}")
    
    def test_knowledge_integration(self):
        """Test integration of knowledge base with solution generation."""
        # Get relevant context for a derivative operation
        context = self.knowledge_base.get_context_for_solution(
            "calculus", "differentiate", "polynomial"
        )
        
        # Verify that context contains relevant knowledge
        self.assertGreater(len(context["concepts"]), 0, "No concepts found in context")
        
        # Test if any of the relevant theorems are found
        found_theorem = False
        for theorem in context["theorems"]:
            if theorem.name in ["Product Rule", "Chain Rule"]:
                found_theorem = True
                break
        
        self.assertTrue(found_theorem, "No relevant calculus theorems found in context")
        
        # Test the explanation enhancement
        explanation = "To find the derivative, we apply the Product Rule."
        enhanced = self.math_agent.enhance_explanation(explanation, {
            "theorems": [t.to_dict() for t in context["theorems"]]
        })
        
        # Verify that the explanation was enhanced
        self.assertGreater(len(enhanced), len(explanation), 
                          "Explanation was not enhanced with knowledge")
    
    def test_agent_message_processing(self):
        """Test the Math Knowledge Agent's message processing capabilities."""
        # Create a computation request message
        request = {
            "header": {
                "message_id": "test_msg_001",
                "timestamp": "2023-10-15T10:30:00Z",
                "message_type": "computation_request",
                "sender": "test_sender",
                "recipient": "math_knowledge_agent"
            },
            "body": {
                "operation": "solve",
                "expression": "x^2 - 4 = 0",
                "domain": "algebra",
                "expression_type": "quadratic_equation"
            }
        }
        
        # Process the message
        response = self.math_agent.process_message(request)
        
        # Verify the response
        self.assertEqual(response["header"]["message_type"], "computation_response", 
                        "Incorrect response message type")
        self.assertTrue(response["body"]["success"], 
                       f"Computation failed: {response['body'].get('error', 'Unknown error')}")
        
        # Verify that context information is included
        self.assertIn("context", response["body"], 
                     "Knowledge context not included in response")
        
        # Create a step solution request
        step_request = {
            "header": {
                "message_id": "test_msg_002",
                "timestamp": "2023-10-15T10:31:00Z",
                "message_type": "step_solution_request",
                "sender": "test_sender",
                "recipient": "math_knowledge_agent"
            },
            "body": {
                "operation": "solve",
                "expression": "x^2 - 4 = 0",
                "domain": "algebra",
                "variables": ["x"]
            }
        }
        
        # Process the step solution request
        step_response = self.math_agent.process_message(step_request)
        
        # Verify the step solution response
        self.assertEqual(step_response["header"]["message_type"], "step_solution_response", 
                        "Incorrect step solution response message type")
        self.assertTrue(step_response["body"]["success"], 
                       f"Step solution failed: {step_response['body'].get('error', 'Unknown error')}")
        self.assertIn("steps", step_response["body"], 
                     "Steps not included in response")
        self.assertGreater(len(step_response["body"]["steps"]), 0, 
                          "No steps generated in response")

if __name__ == "__main__":
    unittest.main()

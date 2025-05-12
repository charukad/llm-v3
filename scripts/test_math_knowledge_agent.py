#!/usr/bin/env python3
"""
Test Script for Math Knowledge Agent

This script demonstrates the enhanced capabilities of the Math Knowledge Agent,
showcasing its integration with the knowledge base, step-by-step solution generation,
and advanced verification mechanisms.
"""

import sys
import os
import json
from datetime import datetime

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_processing.agent.math_knowledge_agent import MathKnowledgeAgent
from math_processing.knowledge.knowledge_base import MathKnowledgeBase

def print_separator(text=""):
    """Print a separator line with optional text."""
    width = 80
    if text:
        text = f" {text} "
        padding = (width - len(text)) // 2
        print("=" * padding + text + "=" * (width - padding - len(text)))
    else:
        print("=" * width)

def print_json(data):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def main():
    """Main function to test the Math Knowledge Agent."""
    print_separator("Math Knowledge Agent Test")
    print("Initializing the Math Knowledge Agent...")
    
    # Initialize the agent
    agent = MathKnowledgeAgent()
    
    # Test 1: Basic computation
    print_separator("Test 1: Basic Computation")
    request = {
        "header": {
            "message_id": "test1",
            "timestamp": datetime.now().isoformat(),
            "message_type": "computation_request",
            "sender": "test_script",
            "recipient": "math_knowledge_agent"
        },
        "body": {
            "operation": "solve",
            "expression": "x^2 - 4 = 0",
            "domain": "algebra",
            "expression_type": "quadratic_equation"
        }
    }
    
    print("Sending computation request:")
    print_json(request)
    
    response = agent.process_message(request)
    
    print("\nReceived response:")
    print_json(response)
    
    # Test 2: Step-by-step solution
    print_separator("Test 2: Step-by-Step Solution")
    request = {
        "header": {
            "message_id": "test2",
            "timestamp": datetime.now().isoformat(),
            "message_type": "step_solution_request",
            "sender": "test_script",
            "recipient": "math_knowledge_agent"
        },
        "body": {
            "operation": "differentiate",
            "expression": "x^2 * sin(x)",
            "variables": ["x"],
            "domain": "calculus"
        }
    }
    
    print("Sending step solution request:")
    print_json(request)
    
    response = agent.process_message(request)
    
    print("\nReceived response:")
    print_json(response)
    
    # Test 3: Solution verification
    print_separator("Test 3: Solution Verification")
    request = {
        "header": {
            "message_id": "test3",
            "timestamp": datetime.now().isoformat(),
            "message_type": "verification_request",
            "sender": "test_script",
            "recipient": "math_knowledge_agent"
        },
        "body": {
            "problem": {
                "problem_type": "derivative",
                "function": "x^3",
                "variable": "x"
            },
            "solution": {
                "derivative": "3*x^2"
            },
            "domain": "calculus"
        }
    }
    
    print("Sending verification request:")
    print_json(request)
    
    response = agent.process_message(request)
    
    print("\nReceived response:")
    print_json(response)
    
    # Test 4: Knowledge query
    print_separator("Test 4: Knowledge Query")
    request = {
        "header": {
            "message_id": "test4",
            "timestamp": datetime.now().isoformat(),
            "message_type": "knowledge_query",
            "sender": "test_script",
            "recipient": "math_knowledge_agent"
        },
        "body": {
            "query_type": "solution_context",
            "domain": "calculus",
            "operation": "differentiate"
        }
    }
    
    print("Sending knowledge query request:")
    print_json(request)
    
    response = agent.process_message(request)
    
    print("\nReceived response:")
    print_json(response)
    
    # Test 5: Explanation enhancement
    print_separator("Test 5: Explanation Enhancement")
    
    kb = MathKnowledgeBase()
    context = kb.get_context_for_solution("calculus", "differentiate", "")
    context_dict = {
        "concepts": [c.to_dict() for c in context["concepts"]],
        "theorems": [t.to_dict() for t in context["theorems"]],
        "formulas": [f.to_dict() for f in context["formulas"]]
    }
    
    explanation = "To find the derivative of this expression, we need to apply the Product Rule and the Chain Rule."
    
    print("Original explanation:")
    print(explanation)
    
    enhanced = agent.enhance_explanation(explanation, context_dict)
    
    print("\nEnhanced explanation:")
    print(enhanced)
    
    print_separator("Tests Completed")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Demonstration of context management for the Mathematical Multimodal LLM System.

This script shows how the context management system works in practice, integrating
with other components and handling a realistic mathematical conversation.
"""

import os
import sys
import json
import time
import random
from typing import Dict, List, Any, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.context.context_manager import get_context_manager
from orchestration.context.entity_tracker import EntityTracker
from orchestration.context.pruning_strategy import TokenBudgetStrategy
from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


def simulate_llm_response(query: str, context: Dict[str, Any]) -> str:
    """
    Simulate an LLM response to a mathematical query.
    
    In a real implementation, this would call the LLM with the query and context.
    Here we use simple pattern matching to generate responses.
    
    Args:
        query: User query
        context: Conversation context
        
    Returns:
        Simulated LLM response
    """
    # Extract relevant entities from context
    relevant_entities = context.get("relevant_entities", [])
    entity_info = ""
    
    if relevant_entities:
        entity_info = "\nRelevant entities found in context:\n"
        for entity in relevant_entities:
            entity_type = entity.get("entity_type", "")
            value = entity.get("value", "")
            display = entity.get("display_form", value)
            entity_info += f"- {entity_type}: {display} = {value}\n"
    
    # Simple pattern matching to generate responses
    if "derivative" in query.lower():
        if "f(x)" in query:
            return f"To find the derivative of f(x), I'll use the rules of differentiation:{entity_info}\n\nGiven that f(x) = x^2 + 3x + 2, the derivative is:\nf'(x) = 2x + 3\n\nThis represents the rate of change of the function at any point x."
        elif "g(x)" in query:
            return f"To find the derivative of g(x), I'll use the rules of differentiation:{entity_info}\n\nGiven that g(x) = sin(x), the derivative is:\ng'(x) = cos(x)\n\nThis represents the rate of change of the sine function at any point x."
        elif "h(x)" in query:
            return f"To find the derivative of h(x), I'll use the rules of differentiation:{entity_info}\n\nGiven that h(x) = e^x, the derivative is:\nh'(x) = e^x\n\nInterestingly, the exponential function is its own derivative!"
        else:
            return f"To find the derivative, I need to know which function you're referring to.{entity_info}\n\nPlease specify the function you'd like me to differentiate."
    
    elif "integrate" in query.lower():
        if "sin(x)" in query:
            return f"To integrate sin(x), I'll use the standard integral:{entity_info}\n\n∫sin(x)dx = -cos(x) + C\n\nWhere C is the constant of integration."
        elif "x^2" in query:
            return f"To integrate x^2, I'll use the power rule for integration:{entity_info}\n\n∫x^2 dx = x^3/3 + C\n\nWhere C is the constant of integration."
        else:
            return f"To perform integration, I need to know which function you're referring to.{entity_info}\n\nPlease specify the function you'd like me to integrate."
    
    elif "roots" in query.lower() or "solve" in query.lower():
        if "f(x)" in query or "first function" in query:
            return f"To find the roots of f(x) = x^2 + 3x + 2, I'll solve the equation f(x) = 0:{entity_info}\n\nx^2 + 3x + 2 = 0\n\nFactoring the quadratic:\n(x + 1)(x + 2) = 0\n\nTherefore:\nx = -1 or x = -2\n\nThe roots of f(x) are x = -1 and x = -2."
        else:
            return f"To find roots or solve an equation, I need to know which function or equation you're referring to.{entity_info}\n\nPlease specify the function or equation you'd like me to work with."
    
    elif "define" in query.lower():
        if "function" in query.lower() and ("f" in query or "f(" in query):
            return f"I'll define the function f(x) = x^2 + 3x + 2. This is a quadratic function where the coefficient of x^2 is 1, the coefficient of x is 3, and the constant term is 2."
        elif "function" in query.lower() and ("g" in query or "g(" in query):
            return f"I'll define the function g(x) = sin(x). This is a trigonometric function with period 2π, which oscillates between -1 and 1."
        elif "function" in query.lower() and ("h" in query or "h(" in query):
            return f"I'll define the function h(x) = e^x. This is the exponential function with base e (Euler's number, approximately 2.71828). It's a fundamental function in calculus with the property that its derivative is itself."
        else:
            return f"I'll define that mathematical entity for you. Please let me know if you'd like to work with it in a particular way."
    
    else:
        return f"I understand you're asking about mathematical concepts.{entity_info}\n\nCan you provide more details about what you'd like me to help with? I can differentiate, integrate, find roots, or perform other mathematical operations."


def run_demo():
    """Run the demonstration of context management."""
    logger.info("Starting context management demonstration")
    
    # Get the context manager
    context_manager = get_context_manager()
    
    # Create a new conversation
    user_id = "demo_user"
    conversation_id = context_manager.create_conversation(user_id)
    
    logger.info(f"Created conversation: {conversation_id}")
    
    # Simulated conversation script
    conversation_script = [
        # Turn 1: Define function f(x)
        "Let's define a function f(x) = x^2 + 3x + 2",
        
        # Turn 2: Find derivative
        "Find the derivative of this function",
        
        # Turn 3: Define another function g(x)
        "Now let's define g(x) = sin(x)",
        
        # Turn 4: Find derivative of g(x)
        "What is the derivative of g(x)?",
        
        # Turn 5: Reference to first function
        "Find the roots of the first function",
        
        # Turn 6: Define a third function
        "Define h(x) = e^x",
        
        # Turn 7: Reference with function name
        "Find the derivative of h(x) and evaluate it at x = 0"
    ]
    
    # Run through the conversation script
    for i, user_query in enumerate(conversation_script):
        logger.info(f"\n--- Turn {i+1} ---")
        logger.info(f"User: {user_query}")
        
        # Add user message to context
        context_manager.add_user_message(conversation_id, user_query)
        
        # Resolve references in the query
        resolved_info = context_manager.resolve_entity_references(conversation_id, user_query)
        resolved_query = resolved_info["resolved_query"]
        
        if resolved_query != user_query:
            logger.info(f"Resolved query: {resolved_query}")
        
        # Get relevant entities
        relevant_entities = context_manager.get_relevant_entities(
            conversation_id, resolved_query)
        
        # Prepare context for the LLM
        llm_context = {
            "resolved_query": resolved_query,
            "relevant_entities": relevant_entities
        }
        
        # Generate LLM response
        llm_response = simulate_llm_response(resolved_query, llm_context)
        logger.info(f"System: {llm_response}")
        
        # Add system response to context
        context_manager.add_system_message(conversation_id, llm_response)
        
        # Get context summary after each turn
        if i == len(conversation_script) - 1:  # Only for the last turn
            summary = context_manager.get_context_summary(conversation_id)
            logger.info("\n--- Context Summary ---")
            logger.info(f"Messages: {summary['message_count']}")
            logger.info(f"Tokens: {summary['token_count']}")
            logger.info(f"Entities: {summary['entity_summary']['total_entities']}")
            logger.info(f"Entity types: {summary['entity_summary']['entity_counts']}")


if __name__ == "__main__":
    run_demo()

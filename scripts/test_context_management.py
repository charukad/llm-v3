#!/usr/bin/env python
"""
Test script for context management components of the Mathematical Multimodal LLM System.

This script demonstrates how to use the context management components in a conversation.
"""

import time
import argparse
import logging
import os
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.context.context_manager import get_context_manager
from orchestration.context.entity_tracker import EntityTracker
from orchestration.context.pruning_strategy import (
    TokenBudgetStrategy, 
    RelevancePruningStrategy
)
from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


def simulate_mathematical_conversation():
    """
    Simulate a mathematical conversation to demonstrate context management.
    
    Returns:
        Conversation summary
    """
    logger.info("Simulating mathematical conversation")
    
    # Get the context manager
    context_manager = get_context_manager()
    
    # Create a new conversation
    user_id = "test_user_123"
    conversation_id = context_manager.create_conversation(user_id)
    
    logger.info(f"Created conversation {conversation_id}")
    
    # Add user and system messages
    messages = [
        # Turn 1: Variable definition
        ("user", "Let's define a function f(x) = x^2 + 3x + 2"),
        ("system", "I've defined the function f(x) = x^2 + 3x + 2. This is a quadratic function."),
        
        # Turn 2: Derivative request
        ("user", "Find the derivative of this function"),
        ("system", "The derivative of f(x) = x^2 + 3x + 2 is f'(x) = 2x + 3."),
        
        # Turn 3: Evaluate at a point
        ("user", "What is the value of the derivative at x = 2?"),
        ("system", "To find the value of f'(x) at x = 2, I'll substitute x = 2 into f'(x) = 2x + 3:\n\nf'(2) = 2(2) + 3 = 4 + 3 = 7\n\nTherefore, the derivative at x = 2 is 7."),
        
        # Turn 4: Another function
        ("user", "Now let's consider g(x) = sin(x)"),
        ("system", "I'll define g(x) = sin(x). This is a trigonometric function with period 2π."),
        
        # Turn 5: Reference to previous function
        ("user", "What is the derivative of this function?"),
        ("system", "The derivative of g(x) = sin(x) is g'(x) = cos(x)."),
        
        # Turn 6: Implicit reference
        ("user", "Let's go back to the first function. Find its roots."),
        ("system", "To find the roots of f(x) = x^2 + 3x + 2, I'll solve f(x) = 0:\n\nx^2 + 3x + 2 = 0\n\nUsing the quadratic formula or factoring:\nx^2 + 3x + 2 = (x + 1)(x + 2) = 0\n\nSo x = -1 or x = -2\n\nThe roots of the function are x = -1 and x = -2."),
        
        # Turn 7: Complex mathematical expression
        ("user", "Define the integral I = ∫(0 to π) sin(x) dx"),
        ("system", "I'll define the integral I as follows:\n\nI = ∫(0 to π) sin(x) dx\n\nCalculating this integral:\nI = [-cos(x)](0 to π)\nI = -cos(π) - (-cos(0))\nI = -(-1) - (-1)\nI = 1 + 1 = 2\n\nTherefore, I = 2."),
        
        # Turn 8: Reference to entity
        ("user", "Now define a new function h(x) = e^x"),
        ("system", "I've defined h(x) = e^x. This is the exponential function with base e."),
        
        # Turn 9: Multiple references
        ("user", "Calculate the derivative of h(x) and evaluate it at x = 0"),
        ("system", "The derivative of h(x) = e^x is h'(x) = e^x.\n\nEvaluating at x = 0:\nh'(0) = e^0 = 1\n\nTherefore, the derivative of h(x) at x = 0 is 1."),
    ]
    
    # Add the messages to the conversation
    for role, content in messages:
        if role == "user":
            result = context_manager.add_user_message(conversation_id, content)
            logger.info(f"Added user message: {content[:30]}... ({result['message_id']})")
        else:
            result = context_manager.add_system_message(conversation_id, content)
            logger.info(f"Added system message: {content[:30]}... ({result['message_id']})")
    
    # Test reference resolution
    references_to_test = [
        "What's the value of the first function at x = 3?",
        "Can you find the second derivative of the last function?",
        "Integrate the function g(x) from 0 to π/2",
        "Compare the derivatives of f(x) and h(x)"
    ]
    
    logger.info("\nTesting reference resolution:")
    for reference in references_to_test:
        resolved = context_manager.resolve_entity_references(conversation_id, reference)
        logger.info(f"  Original: {reference}")
        logger.info(f"  Resolved: {resolved['resolved_query']}")
        logger.info(f"  Referenced entities: {len(resolved['referenced_entities'])}")
    
    # Get conversation context
    context_text = context_manager.get_conversation_context(conversation_id, format="text")
    
    # Get context summary
    summary = context_manager.get_context_summary(conversation_id)
    
    # Test pruning
    logger.info("\nTesting context pruning:")
    logger.info(f"  Before pruning: {summary['token_count']} tokens")
    
    # Create many messages to trigger pruning
    for i in range(10):
        context_manager.add_user_message(
            conversation_id, 
            f"This is message {i+1} to test pruning. It contains some text to increase token count."
        )
        context_manager.add_system_message(
            conversation_id,
            f"Response to message {i+1}. This response also contains additional text to increase the token count and trigger the pruning strategy."
        )
    
    # Get updated summary
    updated_summary = context_manager.get_context_summary(conversation_id)
    logger.info(f"  After adding messages: {updated_summary['token_count']} tokens")
    
    # Get relevant entities for a new query
    query = "Find the derivative of f(x) and evaluate it at x = 1"
    relevant_entities = context_manager.get_relevant_entities(conversation_id, query)
    
    logger.info("\nRelevant entities for query:")
    logger.info(f"  Query: {query}")
    logger.info(f"  Found {len(relevant_entities)} relevant entities")
    
    return {
        "conversation_id": conversation_id,
        "message_count": updated_summary["message_count"],
        "token_count": updated_summary["token_count"],
        "entity_count": updated_summary["entity_summary"]["total_entities"],
        "context_sample": context_text[:200] + "..." if len(context_text) > 200 else context_text
    }


def test_entity_extraction():
    """Test the entity extraction capabilities."""
    logger.info("\nTesting entity extraction:")
    
    # Create an entity tracker
    tracker = EntityTracker()
    
    # Test with various mathematical expressions
    expressions = [
        "Let f(x) = x^2 + 3x + 2",
        "Define the variable y = mx + b",
        "The matrix A = [[1, 2], [3, 4]] is invertible",
        "Consider the integral I = ∫(0 to 1) x^2 dx",
        "Let's use the quadratic formula x = (-b ± √(b^2 - 4ac)) / (2a)",
        "$\\frac{d}{dx}[f(x)] = f'(x)$",
        "The function $g(x) = \\sin(x)$ is periodic"
    ]
    
    for expr in expressions:
        entities = tracker.extract_entities(expr)
        logger.info(f"  Expression: {expr}")
        logger.info(f"  Extracted entities: {len(entities)}")
        
        for entity in entities:
            logger.info(f"    - {entity['entity_type']}: {entity['value']}")


def main():
    """Main function to run context management tests."""
    parser = argparse.ArgumentParser(description="Test context management components")
    parser.add_argument("--tokens", type=int, default=4096,
                      help="Maximum context token limit")
    parser.add_argument("--strategy", choices=["token", "relevance"],
                      default="token", help="Pruning strategy")
    
    args = parser.parse_args()
    
    # Get the context manager
    context_manager = get_context_manager()
    
    # Configure context manager
    context_manager.max_context_tokens = args.tokens
    
    # Set pruning strategy
    if args.strategy == "token":
        context_manager.pruning_strategy = TokenBudgetStrategy(
            max_tokens=args.tokens,
            target_ratio=0.8,
            preserve_system_messages=True,
            preserve_last_n_turns=2
        )
    else:
        context_manager.pruning_strategy = RelevancePruningStrategy(
            max_tokens=args.tokens,
            target_ratio=0.8,
            preserve_last_n_turns=2
        )
    
    logger.info(f"Running context management tests with:")
    logger.info(f"  Token limit: {args.tokens}")
    logger.info(f"  Pruning strategy: {args.strategy}")
    
    # Run entity extraction test
    test_entity_extraction()
    
    # Simulate a conversation
    conversation_summary = simulate_mathematical_conversation()
    
    logger.info("\nConversation Summary:")
    logger.info(f"  Conversation ID: {conversation_summary['conversation_id']}")
    logger.info(f"  Message Count: {conversation_summary['message_count']}")
    logger.info(f"  Token Count: {conversation_summary['token_count']}")
    logger.info(f"  Entity Count: {conversation_summary['entity_count']}")
    logger.info(f"  Context Sample: {conversation_summary['context_sample']}")


if __name__ == "__main__":
    main()

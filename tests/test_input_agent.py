"""
Test script for the Central Input Agent.

This script demonstrates how the Input Agent serves as the central point
of the multimodal system, processing different types of requests and
generating detailed instructions for specialized agents.
"""
import os
import sys
import asyncio
import logging
import json
from typing import Dict, Any

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multimodal.agent.input_agent import get_input_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_input_agent():
    """
    Test the Input Agent with different request types.
    
    This demonstrates how the Input Agent analyzes different types of requests,
    determines the appropriate specialized agent, and generates detailed
    processing instructions.
    """
    # Get the Input Agent instance
    input_agent = get_input_agent()
    
    # Define test cases for different types of requests
    test_cases = [
        {
            "name": "Visualization Request",
            "request": {
                "input_type": "text",
                "content": "Generate a bar chart showing the population of the top 5 most populous countries",
                "parameters": {
                    "prefer_interactive": True,
                }
            },
            "expected_agent": "visualization"
        },
        {
            "name": "Math Computation Request",
            "request": {
                "input_type": "text",
                "content": "Solve the system of equations: 2x + y = 10, 3x - 2y = 5",
                "parameters": {
                    "show_steps": True
                }
            },
            "expected_agent": "math_computation"
        },
        {
            "name": "General Knowledge Request",
            "request": {
                "input_type": "text",
                "content": "What are the advantages and disadvantages of renewable energy sources?",
                "parameters": {}
            },
            "expected_agent": "core_llm"
        },
        {
            "name": "Image Analysis with Math",
            "request": {
                "input_type": "image",
                "content": {
                    "recognized_latex": "\\int_0^\\pi \\sin(x) dx"
                },
                "parameters": {}
            },
            "expected_agent": "math_computation"
        }
    ]
    
    # Process each test case
    for i, test_case in enumerate(test_cases):
        logger.info(f"\n{'='*80}\nTesting Case {i+1}: {test_case['name']}\n{'='*80}")
        
        # Define a test conversation ID
        conversation_id = f"test-conversation-{i+1}"
        
        # Process the request through the Input Agent
        result = await input_agent.process_request(
            request_data=test_case["request"],
            conversation_id=conversation_id
        )
        
        # Print the results
        logger.info(f"Request processed with ID: {result.get('request_id')}")
        logger.info(f"Workflow ID: {result.get('workflow_id')}")
        logger.info(f"Detected agent type: {result.get('agent_type')}")
        logger.info(f"Instructions summary: {result.get('instructions_summary')}")
        
        # Check if the result matches the expected agent
        if result.get("agent_type") == test_case["expected_agent"]:
            logger.info(f"✅ Test passed: Correctly routed to {result.get('agent_type')}")
        else:
            logger.warning(f"❌ Test failed: Expected {test_case['expected_agent']}, got {result.get('agent_type')}")
        
        logger.info(f"Processing time: {result.get('processing_time_ms', 0):.2f} ms")
        logger.info("-" * 80)
    
    # Test explicit agent routing
    logger.info("\n" + "="*80)
    logger.info("Testing explicit agent routing")
    logger.info("="*80)
    
    # Create a request with an explicit target agent
    explicit_request = {
        "input_type": "text",
        "content": "What is the capital of France?",
        "parameters": {}
    }
    
    # Process with explicit agent routing
    result = await input_agent.process_request(
        request_data={
            **explicit_request,
            "parameters": {"target_agent": "search"}
        },
        conversation_id="test-explicit-routing"
    )
    
    logger.info(f"Request processed with ID: {result.get('request_id')}")
    logger.info(f"Explicitly routed to: {result.get('agent_type')}")
    logger.info(f"Instructions summary: {result.get('instructions_summary')}")
    
    if result.get("agent_type") == "search":
        logger.info("✅ Test passed: Correctly respected explicit agent routing")
    else:
        logger.warning(f"❌ Test failed: Expected search, got {result.get('agent_type')}")

if __name__ == "__main__":
    asyncio.run(test_input_agent()) 
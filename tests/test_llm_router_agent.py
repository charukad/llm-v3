"""
Test script for the LLM Router Agent.

This script tests the LLM Router Agent's ability to analyze different types of inputs
and route them to the appropriate specialized agents.
"""
import os
import sys
import asyncio
import logging
import json
from typing import Dict, Any

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multimodal.agent.llm_router_agent import LLMRouterAgent
from multimodal.unified_pipeline.content_router import ContentRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_router_agent():
    """Test the LLM Router Agent with different input types."""
    router_agent = LLMRouterAgent()
    
    # Test cases
    test_inputs = [
        {
            "name": "Simple math expression",
            "input": {
                "input_type": "text",
                "text": "What is the derivative of sin(x^2)?",
                "success": True,
                "text_type": "plain"
            },
            "expected_agent": "math_computation"
        },
        {
            "name": "Question about history",
            "input": {
                "input_type": "text",
                "text": "Who was Albert Einstein and what were his contributions to physics?",
                "success": True,
                "text_type": "plain"
            },
            "expected_agent": "core_llm"
        },
        {
            "name": "Visualization request",
            "input": {
                "input_type": "text",
                "text": "Plot a 3D visualization of the function z = sin(x) * cos(y)",
                "success": True,
                "text_type": "plain"
            },
            "expected_agent": "visualization"
        },
        {
            "name": "OCR-like input",
            "input": {
                "input_type": "image",
                "image_type": "image/png",
                "recognized_latex": "\\int_{0}^{\\pi} \\sin(x) dx",
                "success": True
            },
            "expected_agent": "math_computation"
        }
    ]
    
    for test_case in test_inputs:
        logger.info(f"Testing: {test_case['name']}")
        
        # Process the input with the router agent
        result = router_agent.route_request(test_case["input"])
        
        # Check if the routing was successful
        if result["success"]:
            routing_decision = result["routing_decision"]
            primary_agent = routing_decision["primary_agent"]
            confidence = routing_decision["confidence"]
            capabilities = routing_decision["capabilities_needed"]
            reasoning = routing_decision["reasoning"]
            
            logger.info(f"  -> Routed to: {primary_agent} (confidence: {confidence:.2f})")
            logger.info(f"  -> Capabilities: {capabilities}")
            logger.info(f"  -> Reasoning: {reasoning}")
            
            if primary_agent == test_case["expected_agent"]:
                logger.info(f"  ✓ Test passed: Correctly routed to {primary_agent}")
            else:
                logger.warning(f"  ✗ Test failed: Expected {test_case['expected_agent']}, got {primary_agent}")
                logger.warning(f"    Reason: {reasoning}")
        else:
            logger.error(f"  ✗ Test failed: Routing was unsuccessful - {result.get('error', 'Unknown error')}")
            
        logger.info("-" * 50)
    
    # Test with content router integration
    logger.info("Testing ContentRouter with LLM integration")
    content_router = ContentRouter({"use_llm_router": True})
    
    for test_case in test_inputs:
        logger.info(f"ContentRouter Test: {test_case['name']}")
        
        # Process the input with the content router
        result = content_router.route_content(test_case["input"])
        
        agent_type = result.get("agent_type")
        source_type = result.get("source_type")
        
        logger.info(f"  -> ContentRouter selected agent: {agent_type}")
        logger.info(f"  -> Source type: {source_type}")
        
        if agent_type and source_type:
            logger.info(f"  ✓ ContentRouter integration test passed")
        else:
            logger.error(f"  ✗ ContentRouter integration test failed: {result}")
            
        logger.info("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_router_agent()) 
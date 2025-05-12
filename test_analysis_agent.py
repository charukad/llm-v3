#!/usr/bin/env python3
"""
Test script for the MathAnalysisAgent.
"""
import os
import sys
import json
import asyncio
import logging

# Add the project root to the Python path to fix imports
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now we can import project modules
from core.agent.analysis_agent import MathAnalysisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("analysis_agent_test")

class MockLLMAgent:
    """Mock LLM agent for testing."""
    
    def __init__(self, response=None):
        self.response = response or {
            "operations": ["equation_solving"],
            "concepts": ["algebra", "quadratic_equations"],
            "required_agents": ["core_llm_agent", "math_computation_agent"],
            "complexity": "simple",
            "sub_problems": [],
            "routing": {
                "primary_agent": "math_computation_agent",
                "confidence": 0.85,
                "alternative_agents": ["core_llm_agent"]
            }
        }
    
    def generate_text(self, prompt, **kwargs):
        """Mock generation method."""
        return json.dumps(self.response, indent=2)
    
    def generate(self, prompt, **kwargs):
        """Mock generation method."""
        return json.dumps(self.response, indent=2)
    
    def __call__(self, prompt):
        """Mock call method."""
        return json.dumps(self.response, indent=2)

async def test_analysis_agent():
    """Test the MathAnalysisAgent with a mock LLM."""
    logger.info("Testing MathAnalysisAgent with mock LLM")
    
    # Create a mock LLM agent
    mock_llm = MockLLMAgent()
    
    # Create the analysis agent with the mock LLM
    analysis_agent = MathAnalysisAgent(llm_agent=mock_llm)
    
    # Test queries
    test_queries = [
        "Solve the quadratic equation x^2 + 5x + 6 = 0",
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        "Plot the function y = sin(x) from -π to π"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        result = await analysis_agent.analyze_query(query)
        logger.info(f"Analysis result: {json.dumps(result, indent=2)}")
        logger.info("-" * 50)
    
    # Test with a custom response
    logger.info("Testing with a custom response for calculus")
    calculus_response = {
        "operations": ["differentiation"],
        "concepts": ["calculus", "derivatives"],
        "required_agents": ["core_llm_agent", "math_computation_agent"],
        "complexity": "moderate",
        "sub_problems": [
            {"type": "rule_application", "description": "Apply power rule"}
        ],
        "routing": {
            "primary_agent": "math_computation_agent",
            "confidence": 0.9,
            "alternative_agents": ["core_llm_agent"]
        }
    }
    
    # Create a new mock with the custom response
    mock_llm_calculus = MockLLMAgent(calculus_response)
    analysis_agent.llm_agent = mock_llm_calculus
    
    calculus_query = "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1"
    logger.info(f"Testing query with custom response: {calculus_query}")
    result = await analysis_agent.analyze_query(calculus_query)
    logger.info(f"Analysis result: {json.dumps(result, indent=2)}")

    # Test with a fallback case (no LLM)
    logger.info("Testing fallback case (no LLM)")
    analysis_agent.llm_agent = None
    fallback_query = "Integrate x^2 from 0 to 1"
    logger.info(f"Testing query with fallback: {fallback_query}")
    result = await analysis_agent.analyze_query(fallback_query)
    logger.info(f"Fallback analysis result: {json.dumps(result, indent=2)}")
    
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(test_analysis_agent()) 
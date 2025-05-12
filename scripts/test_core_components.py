"""
Test script for core components of the Mathematical Multimodal LLM System.

This script tests the basic functionality of the Core LLM Agent and
mathematical processing components.
"""

import logging
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_symbolic_processor():
    """Test the Symbolic Processor."""
    logger.info("Testing Symbolic Processor...")
    
    try:
        # Import here to avoid circular imports
        from math_processing.computation.sympy_wrapper import SymbolicProcessor
        
        # Test equation solving
        logger.info("Testing equation solving...")
        solutions = SymbolicProcessor.solve_equation("x^2 - 5*x + 6 = 0", "x")
        logger.info(f"Solutions: {solutions}")
        
        # Test differentiation
        logger.info("Testing differentiation...")
        derivative = SymbolicProcessor.differentiate("x^3 + 2*x^2 - 5*x + 3", "x")
        logger.info(f"Derivative: {derivative}")
        
        # Test integration
        logger.info("Testing integration...")
        integral = SymbolicProcessor.integrate("x^2", "x")
        logger.info(f"Indefinite integral: {integral}")
        
        logger.info("Symbolic Processor tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Symbolic Processor test failed: {e}")
        return False

def test_basic_llm_response():
    """Test basic response generation from the LLM."""
    logger.info("Testing basic LLM response generation...")
    
    try:
        # Import here to avoid circular imports
        from core.agent.llm_agent import CoreLLMAgent
        
        # Initialize agent
        agent = CoreLLMAgent()
        
        # Test with a very simple query
        query = "What is 2+2?"
        logger.info(f"Generating response for: '{query}'")
        
        # Generate response with minimal parameters
        response = agent.generate_response(
            prompt=query,
            system_prompt="You are a helpful assistant.",
            max_new_tokens=20
        )
        
        logger.info(f"Response: '{response}'")
        logger.info("Basic LLM response test completed")
        return True
        
    except Exception as e:
        logger.error(f"Basic LLM response test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting core component tests...")
    
    # Test each component independently
    symbolic_test_success = test_symbolic_processor()
    logger.info("-----------------------------------")
    
    llm_test_success = test_basic_llm_response()
    
    # Report results
    logger.info("===================================")
    logger.info("Test Results:")
    logger.info(f"Symbolic Processor: {'PASSED' if symbolic_test_success else 'FAILED'}")
    logger.info(f"Basic LLM Response: {'PASSED' if llm_test_success else 'FAILED'}")
    
    if symbolic_test_success and llm_test_success:
        logger.info("All tests completed successfully")
        sys.exit(0)
    else:
        logger.error("One or more tests failed")
        sys.exit(1)

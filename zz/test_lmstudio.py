#!/usr/bin/env python3
"""
Test script for the LMStudio inference integration.

This script tests the LMStudio API integration with the Mathematical Multimodal LLM system.
"""
import logging
import os
import sys
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from core.mistral.inference import InferenceEngine, LMStudioInference
from core.agent.llm_agent import CoreLLMAgent
from core.prompting.system_prompts import get_system_prompt

def test_lmstudio_direct():
    """Test direct connection to LMStudio API."""
    logger.info("Testing direct connection to LMStudio API")
    
    # Configuration
    lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
    lmstudio_model = os.environ.get("LMSTUDIO_MODEL", "mistral-7b-instruct-v0.3")
    
    try:
        # Create LMStudio inference client
        inference = LMStudioInference(
            api_url=lmstudio_url,
            model_name=lmstudio_model
        )
        
        # Test prompts
        test_prompts = [
            "Explain the Pythagorean theorem in simple terms.",
            "Calculate the derivative of f(x) = x^3 + 2x^2 - 5x + 7",
            "Solve the equation: 3x^2 - 12 = 0"
        ]
        
        for prompt in test_prompts:
            logger.info(f"Testing prompt: {prompt}")
            
            # Generate response
            start_time = time.time()
            response = inference.generate(prompt, max_tokens=150)
            gen_time = time.time() - start_time
            
            # Log results
            logger.info(f"Response generated in {gen_time:.2f} seconds")
            logger.info(f"Response: {response[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error in direct LMStudio test: {e}")
        return False

def test_inference_engine():
    """Test the InferenceEngine with LMStudio configuration."""
    logger.info("Testing InferenceEngine with LMStudio integration")
    
    # Configuration
    lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
    lmstudio_model = os.environ.get("LMSTUDIO_MODEL", "mistral-7b-instruct-v0.3")
    
    try:
        # Create inference engine
        inference = InferenceEngine(
            model_path="placeholder",  # Not used with LMStudio
            use_lmstudio=True,
            lmstudio_url=lmstudio_url,
            lmstudio_model=lmstudio_model
        )
        
        # Test with a mathematical prompt
        prompt = "What are the applications of linear algebra in machine learning?"
        logger.info(f"Testing prompt: {prompt}")
        
        # Generate response
        start_time = time.time()
        response = inference.generate(prompt, max_tokens=200)
        gen_time = time.time() - start_time
        
        # Log results
        logger.info(f"Response generated in {gen_time:.2f} seconds")
        logger.info(f"Response: {response[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error in InferenceEngine test: {e}")
        return False

def test_core_llm_agent():
    """Test the CoreLLMAgent with LMStudio integration."""
    logger.info("Testing CoreLLMAgent with LMStudio integration")
    
    # Configuration
    lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
    lmstudio_model = os.environ.get("LMSTUDIO_MODEL", "mistral-7b-instruct-v0.3")
    
    try:
        # Create core LLM agent with LMStudio configuration
        config = {
            "use_lmstudio": True,
            "lmstudio_url": lmstudio_url,
            "lmstudio_model": lmstudio_model
        }
        
        agent = CoreLLMAgent(config)
        
        # Test mathematical domains
        test_domains = ["algebra", "calculus", "statistics"]
        test_prompt = "Explain the fundamental theorem of calculus"
        
        for domain in test_domains:
            logger.info(f"Testing with {domain} system prompt")
            
            # Get domain-specific system prompt
            system_prompt = get_system_prompt(domain)
            
            # Generate response
            start_time = time.time()
            result = agent.generate_response(test_prompt, system_prompt)
            gen_time = time.time() - start_time
            
            # Check success
            if result.get("success", False):
                logger.info(f"Response generated in {gen_time:.2f} seconds")
                logger.info(f"Response: {result['response'][:100]}...")
            else:
                logger.error(f"Failed to generate response: {result.get('error')}")
                return False
        
        # Test mathematical classification
        logger.info("Testing mathematical domain classification")
        classification = agent.classify_mathematical_domain("Find the integral of sin(x) dx")
        
        if classification.get("success", False):
            logger.info(f"Classified as: {classification['domain']} (confidence: {classification['confidence']})")
        else:
            logger.error(f"Classification failed: {classification.get('error')}")
        
        return True
    except Exception as e:
        logger.error(f"Error in CoreLLMAgent test: {e}")
        return False

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test LMStudio integration")
    parser.add_argument("--url", help="LMStudio API URL", default="http://127.0.0.1:1234")
    parser.add_argument("--model", help="Model name in LMStudio", default="mistral-7b-instruct-v0.3")
    parser.add_argument("--tests", choices=["all", "direct", "engine", "agent"], default="all",
                      help="Which tests to run")
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["LMSTUDIO_URL"] = args.url
    os.environ["LMSTUDIO_MODEL"] = args.model
    os.environ["USE_LMSTUDIO"] = "1"
    
    logger.info(f"Testing LMStudio integration with {args.model} at {args.url}")
    
    results = {}
    
    # Run selected tests
    if args.tests in ["all", "direct"]:
        results["direct"] = test_lmstudio_direct()
    
    if args.tests in ["all", "engine"]:
        results["engine"] = test_inference_engine()
    
    if args.tests in ["all", "agent"]:
        results["agent"] = test_core_llm_agent()
    
    # Print summary
    logger.info("=== Test Results ===")
    for test, passed in results.items():
        logger.info(f"{test}: {'✅ PASSED' if passed else '❌ FAILED'}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script for Mistral 7B integration.

This script tests the integration of Mistral 7B with the Core LLM Agent,
checking basic functionality and evaluating mathematical capabilities.
"""
import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent.llm_agent import CoreLLMAgent
from core.evaluation.math_evaluator import MathEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_mistral_integration.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Mistral 7B integration')
    
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Model cache directory')
    parser.add_argument('--quantization', type=str, default='4bit',
                        choices=['4bit', '8bit', 'gptq', 'none'],
                        help='Quantization method to use')
    parser.add_argument('--use-vllm', action='store_true',
                        help='Use vLLM for inference if available')
    parser.add_argument('--test-domains', type=str, nargs='+',
                        default=['algebra', 'calculus'],
                        help='Mathematical domains to test')
    parser.add_argument('--max-examples', type=int, default=2,
                        help='Maximum examples per domain to evaluate')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run a quick smoke test without full evaluation')
    
    return parser.parse_args()

def test_basic_functionality(agent: CoreLLMAgent) -> bool:
    """
    Test basic functionality of the Core LLM Agent.
    
    Args:
        agent: Core LLM Agent to test
        
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing basic functionality...")
    
    try:
        # Test 1: Generate a response to a simple question
        response = agent.generate_response(
            prompt="What is 2 + 2?",
            temperature=0.1
        )
        logger.info(f"Response to '2 + 2': {response[:100]}...")
        
        # Test 2: Test domain classification
        classification = agent.classify_mathematical_domain("Find the derivative of x^2")
        logger.info(f"Domain classification: {classification['domain']} with confidence {classification['confidence']:.2f}")
        
        # Test 3: Test expression parsing
        expression = agent.parse_mathematical_expression("Solve for x: 2x + 3 = 7")
        logger.info(f"Parsed expression: {expression}")
        
        logger.info("Basic functionality tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False

def run_smoke_test(agent: CoreLLMAgent) -> bool:
    """
    Run a smoke test with a single query.
    
    Args:
        agent: Core LLM Agent to test
        
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Running smoke test...")
    
    try:
        query = "Solve the equation: 3x + 2 = 8"
        
        classification = agent.classify_mathematical_domain(query)
        logger.info(f"Domain classification: {classification['domain']} with confidence {classification['confidence']:.2f}")
        
        response = agent.generate_response(
            prompt=query,
            domain=classification["domain"],
            use_cot=True,
            num_examples=1
        )
        
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")
        
        logger.info("Smoke test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Configure agent
    agent_config = {
        "model_cache_dir": args.cache_dir,
        "use_quantization": args.quantization != "none",
        "quantization_method": args.quantization if args.quantization != "none" else None,
        "use_vllm": args.use_vllm,
        "temperature": 0.1,
        "max_tokens": 2048
    }
    
    # Initialize agent
    logger.info("Initializing Core LLM Agent...")
    agent = CoreLLMAgent(config=agent_config)
    
    # Test basic functionality
    if not test_basic_functionality(agent):
        logger.error("Basic functionality test failed, exiting.")
        sys.exit(1)
    
    # Run smoke test if requested
    if args.smoke_test:
        if run_smoke_test(agent):
            logger.info("Smoke test passed, exiting.")
            sys.exit(0)
        else:
            logger.error("Smoke test failed, exiting.")
            sys.exit(1)
    
    # Initialize evaluator
    logger.info("Initializing Math Evaluator...")
    evaluator = MathEvaluator(agent)
    
    # Run evaluation
    logger.info(f"Running evaluation on domains: {args.test_domains}")
    results = {}
    
    for domain in args.test_domains:
        domain_results = evaluator.evaluate_domain(domain, args.max_examples)
        results[domain] = domain_results
    
    # Calculate overall results
    total_score = sum(r["average_score"] * r["num_examples"] for r in results.values() if "error" not in r)
    total_examples = sum(r["num_examples"] for r in results.values() if "error" not in r)
    overall_score = total_score / total_examples if total_examples > 0 else 0
    
    evaluation_summary = {
        "overall_score": overall_score,
        "total_examples": total_examples,
        "domains_evaluated": len(results),
        "domain_results": results
    }
    
    logger.info(f"Evaluation complete. Overall score: {overall_score:.2f} across {total_examples} examples")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    evaluator.save_results(evaluation_summary, output_path)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

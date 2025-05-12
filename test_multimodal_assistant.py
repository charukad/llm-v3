#!/usr/bin/env python3
"""
Test script for the MultimodalAssistant class.
Demonstrates processing different types of queries: text, math, and plot.
"""

import os
import logging
import argparse
import time
from multimodal_assistant import MultimodalAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multimodal_assistant_test.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the MultimodalAssistant with different queries")
    parser.add_argument("--model-dir", type=str, default="models/mistral-7b-instruct",
                        help="Directory containing the model")
    parser.add_argument("--model-file", type=str, default="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                        help="Name of the model file")
    parser.add_argument("--context-length", type=int, default=2048,
                        help="Context length for the model")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads to use")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for inference")
    parser.add_argument("--gpu-layers", type=int, default=0,
                        help="Number of layers to offload to GPU (0 for CPU-only)")
    parser.add_argument("--query-type", type=str, choices=["text", "math", "plot", "all"],
                        default="all", help="Type of query to test")
    
    return parser.parse_args()

def main():
    """Main function to run the test."""
    args = parse_args()
    
    # Create the assistant
    assistant = MultimodalAssistant(
        model_dir=args.model_dir,
        model_file=args.model_file,
        context_length=args.context_length,
        num_threads=args.threads,
        batch_size=args.batch_size,
        gpu_layers=args.gpu_layers
    )
    
    # Test the assistant with different queries
    test_queries = {
        "text": "Explain the concept of quantum computing in simple terms.",
        "math": "Calculate the derivative of f(x) = 3x^2 + 2x - 5.",
        "plot": "Create a scatter plot showing the relationship between x and y = x^2 for x in range(-10, 10)."
    }
    
    if args.query_type == "all":
        query_types_to_test = list(test_queries.keys())
    else:
        query_types_to_test = [args.query_type]
    
    for query_type in query_types_to_test:
        logger.info(f"\n\n{'='*50}")
        logger.info(f"Testing {query_type.upper()} query")
        logger.info(f"{'='*50}")
        
        query = test_queries[query_type]
        logger.info(f"Query: {query}")
        
        start_time = time.time()
        result = assistant.process_query(query=query, query_type=query_type)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Response: {result}")
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")
        
        if query_type == "plot" and isinstance(result, str) and result.startswith("file://"):
            logger.info(f"Plot generated at: {result}")
    
    logger.info("\nAll tests completed!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script for the MultimodalAssistant class with sample prompts for each mode.
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
        logging.FileHandler('multimodal_samples_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Sample prompts for each query type
SAMPLE_PROMPTS = {
    "text": [
        "Explain how machine learning algorithms work for image recognition",
        "Describe the process of photosynthesis in plants",
        "What are the main differences between renewable and non-renewable energy sources?",
        "How do neural networks process language?",
        "Explain the concept of blockchain in simple terms"
    ],
    "math": [
        "Calculate the derivative of f(x) = x^3 - 4x^2 + 7x - 2",
        "Solve the equation 3x^2 - 12 = 0",
        "What is the integral of sin(x)*cos(x)?",
        "Find the eigenvalues of the matrix [[4, 2], [1, 3]]",
        "Calculate the limit of (1 + 1/n)^n as n approaches infinity"
    ],
    "plot": [
        "Create a line plot of the function f(x) = sin(x) for x in range(-3.14, 3.14)",
        "Plot y = x^2 and y = 2x on the same graph for x in range(-5, 5)",
        "Generate a scatter plot of 20 random points with a line of best fit",
        "Create a bar chart showing the population of 5 major cities",
        "Plot a 3D surface for z = sin(sqrt(x^2 + y^2)) for x,y in range(-5, 5)"
    ]
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the MultimodalAssistant with sample prompts")
    parser.add_argument("--model-dir", type=str, default="models/mistral-7b-instruct",
                      help="Directory containing the model")
    parser.add_argument("--model-file", type=str, default="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                      help="Name of the model file")
    parser.add_argument("--context-length", type=int, default=4096,
                      help="Context length for the model")
    parser.add_argument("--threads", type=int, default=4,
                      help="Number of threads to use")
    parser.add_argument("--batch-size", type=int, default=512,
                      help="Batch size for inference")
    parser.add_argument("--gpu-layers", type=int, default=0,
                      help="Number of layers to offload to GPU (0 for CPU-only)")
    parser.add_argument("--query-type", type=str, choices=["text", "math", "plot", "all"],
                      default="all", help="Type of query to test")
    parser.add_argument("--sample-index", type=int, default=0,
                      help="Index of the sample prompt to use (0-4)")
    parser.add_argument("--max-tokens", type=int, default=300,
                      help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for generation")
    
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
        gpu_layers=128
    )
    
    # Determine which query types to test
    if args.query_type == "all":
        query_types_to_test = ["text", "math", "plot"]
    else:
        query_types_to_test = [args.query_type]
    
    # Ensure the sample index is valid
    sample_index = min(args.sample_index, 4)
    
    # Test each query type
    for query_type in query_types_to_test:
        logger.info(f"\n\n{'='*50}")
        logger.info(f"Testing {query_type.upper()} query")
        logger.info(f"{'='*50}")
        
        query = SAMPLE_PROMPTS[query_type][sample_index]
        logger.info(f"Query: {query}")
        
        start_time = time.time()
        
        # Process the query
        result = assistant.process_query(
            query=query, 
            query_type=query_type,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        elapsed_time = time.time() - start_time
        
        # For console output
        print("\n" + "="*70)
        print(f"QUERY TYPE: {query_type}")
        print(f"QUERY: {query}")
        print("="*70)
        print(f"\nRESPONSE:\n{result}")
        print("\n" + "="*70)
        print(f"Time taken: {elapsed_time:.2f} seconds")
        
        # For logging
        logger.info(f"Response: {result}")
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")
        
        # Handle plot results
        if query_type == "plot" and isinstance(result, str) and result.startswith("file://"):
            logger.info(f"Plot generated at: {result}")
            print(f"Plot generated at: {result}")
    
    logger.info("\nAll tests completed!")

if __name__ == "__main__":
    main() 
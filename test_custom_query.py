#!/usr/bin/env python3
"""
Test script for the MultimodalAssistant class with a custom query.
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
        logging.FileHandler('custom_query_test.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the MultimodalAssistant with a custom query")
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
    parser.add_argument("--query-type", type=str, choices=["text", "math", "plot"],
                      default="text", help="Type of query to test")
    parser.add_argument("--custom-query", type=str, required=True,
                      help="Custom query to test with the model")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=500,
                      help="Maximum tokens to generate")
    
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
    
    logger.info(f"\n\n{'='*50}")
    logger.info(f"Testing CUSTOM {args.query_type.upper()} query")
    logger.info(f"{'='*50}")
    
    query = args.custom_query
    logger.info(f"Query: {query}")
    
    start_time = time.time()
    
    # Process the query based on the type
    result = assistant.process_query(
        query=query, 
        query_type=args.query_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    elapsed_time = time.time() - start_time
    
    # For console output
    print("\n" + "="*70)
    print(f"QUERY: {query}")
    print("="*70)
    print(f"\nRESPONSE:\n{result}")
    print("\n" + "="*70)
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # For logging
    logger.info(f"Response: {result}")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Handle plot results
    if args.query_type == "plot" and isinstance(result, str) and result.startswith("file://"):
        logger.info(f"Plot generated at: {result}")
        print(f"Plot generated at: {result}")
    
    logger.info("\nTest completed!")

if __name__ == "__main__":
    main() 
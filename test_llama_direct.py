#!/usr/bin/env python3
"""
Direct test of llama-cpp-python without the abstraction layers.
"""
import time
import json
import sys
import logging
from llama_cpp import Llama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_llama_direct(query):
    """Test llama-cpp-python directly."""
    
    # Path to model
    model_path = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    logger.info(f"Loading model from {model_path}")
    start_time = time.time()
    
    # Initialize llama_cpp model directly
    model = Llama(
        model_path=model_path,
        n_ctx=512,  # Smaller context
        n_threads=4,  # Explicit thread count
        n_gpu_layers=32  # GPU layers
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Simple prompt
    prompt = f"<s>[INST] {query} [/INST]"
    
    logger.info(f"Starting generation for prompt: {prompt}")
    start_time = time.time()
    
    # Generate with increased tokens and timeout
    output = model(
        prompt,
        max_tokens=256,
        temperature=0.1,
        echo=False,
        stream=False,
        timeout_ms=15000  # 15 second timeout
    )
    
    gen_time = time.time() - start_time
    logger.info(f"Generation completed in {gen_time:.2f} seconds")
    
    # Extract generated text
    generated_text = output["choices"][0]["text"].strip()
    logger.info(f"Generated text: {generated_text}")
    
    return {
        "success": True,
        "response": generated_text,
        "load_time": load_time,
        "generation_time": gen_time
    }

if __name__ == "__main__":
    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 22323+22?"
    
    logger.info(f"Running test with query: {query}")
    
    result = test_llama_direct(query)
    
    if result.get("success", False):
        print("\nSUCCESS: Direct Llama test completed")
        print(f"Response: {result.get('response')}")
        print(f"Load time: {result.get('load_time'):.2f}s")
        print(f"Generation time: {result.get('generation_time'):.2f}s")
    else:
        print("\nFAILURE: Could not complete direct Llama test")
        print(f"Error: {result.get('error', 'Unknown error')}") 
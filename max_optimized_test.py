#!/usr/bin/env python3
"""
Maximally optimized script for testing the Mistral GGUF model with Metal GPU acceleration.
"""

import os
import time
import logging
from llama_cpp import Llama

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Existing model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Define the prompt
PROMPT = "Calculate the derivative of f(x) = 3x^2 + 2x - 5 can you give me the answer in a step by step manner"

def run_optimized_test():
    """Run a highly optimized test with the existing Mistral model."""
    # Start timing
    start_time = time.time()
    
    print(f"Loading model from {MODEL_PATH}...")
    print("Using maximum performance optimizations with GPU acceleration...")
    
    # Initialize the model with extreme optimization settings
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-100,     # Use all possible layers on GPU
        main_gpu=0,          # Use primary GPU
        n_ctx=1280,           # Tiny context window for speed
        n_threads=1,         # Single thread since we're using GPU
        n_batch=512,         # Large batch size
        use_mlock=True,      # Lock memory
        use_mmap=True,       # Use memory mapping
        verbose=False,       # No verbosity
        seed=42             # Fixed seed for deterministic results
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Setup minimal prompt
    # Keep prompt very short to minimize tokens
    print(f"\nSending prompt: {PROMPT}")
    print("Generating response...")
    
    # Generate without streaming for faster results
    generation_start = time.time()
    
    response = llm(
        PROMPT,              # Minimal prompt
        max_tokens=1000,       # Very limited tokens for faster generation
        temperature=0.0,     # Zero temperature for deterministic output
        top_p=0.95,
        repeat_penalty=1.0,  # No penalty
        echo=False
    )
    
    generation_time = time.time() - generation_start
    
    # Output results
    print("\n--- MODEL RESPONSE ---\n")
    print(response["choices"][0]["text"].strip())
    print("\n---------------------\n")
    print(f"Response generated in {generation_time:.2f} seconds")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
if __name__ == "__main__":
    run_optimized_test() 
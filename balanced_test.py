#!/usr/bin/env python3
"""
Balanced optimization script for Mistral model that provides
reasonable speed while ensuring complete responses.
"""

import os
import time
import logging
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def run_balanced_test(prompt="Calculate the derivative of f(x) = 3x^2 + 2x - 5"):
    """Run a balanced test with speed and completeness in mind."""
    start_time = time.time()
    
    print(f"Loading model from {MODEL_PATH}...")
    print("Using balanced optimizations (speed + complete answers)...")
    
    # Balanced model settings
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=24,      # Use part of the model on GPU to balance memory
        main_gpu=0,           # Use primary GPU
        n_ctx=256,            # Moderate context size
        n_threads=2,          # Two threads for CPU operations
        n_batch=256,          # Moderate batch size
        use_mlock=True,       # Lock memory
        use_mmap=True,        # Use memory mapping
        verbose=False,        # No verbosity
        seed=42              # Fixed seed
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    print(f"\nSending prompt: {prompt}")
    print("Generating response...")
    
    # Generate with balanced settings
    generation_start = time.time()
    
    # Clear and focused prompt with system instructions
    full_prompt = f"""<s>[INST] You are a mathematical assistant. Answer the following question step by step, being clear but concise. Show each step of your work and the final result.

{prompt} [/INST]
"""
    
    response = llm(
        full_prompt,
        max_tokens=200,      # Reasonable token limit
        temperature=0.1,     # Low temperature for more deterministic output
        top_p=0.9,           # Slight filtering
        repeat_penalty=1.1,  # Slight penalty to prevent loops
        echo=False
    )
    
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time
    
    # Output results
    print("\n--- MODEL RESPONSE ---\n")
    print(response["choices"][0]["text"].strip())
    print("\n---------------------\n")
    print(f"Response generated in {generation_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    run_balanced_test() 
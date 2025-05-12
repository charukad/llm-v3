#!/usr/bin/env python3
"""
Final optimized script for Mistral model that balances speed and response quality.
This version focuses on complete mathematical derivatives with maximum performance.
"""

import os
import time
import logging
from llama_cpp import Llama

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def derivative_calculator(expression="3x^2 + 2x - 5"):
    """Calculate the derivative of the given expression with optimized settings."""
    start_time = time.time()
    
    print(f"Loading model from {MODEL_PATH}...")
    print("Using final optimized settings...")
    
    # Optimized model with balanced settings
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=20,       # Partial GPU acceleration
        n_ctx=128,             # Small context size for speed
        n_threads=1,           # Single CPU thread
        n_batch=256,           # Moderate batch size
        use_mlock=True,        # Lock memory
        use_mmap=True,         # Use memory mapping
        verbose=False          # No verbosity
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Create a highly focused prompt specifically for calculus/derivatives
    # This focuses the model to generate more direct and faster responses
    math_prompt = f"""<s>[INST] You are a calculus expert. Calculate the derivative of f(x) = {expression}.
Show your work using the power rule, chain rule, etc. as needed.
Be clear but concise. Show only the essential steps.
Give the final answer in simplified form. [/INST]
"""
    
    print(f"\nCalculating derivative of: f(x) = {expression}")
    print("Generating response...")
    
    # Generation with balanced settings
    generation_start = time.time()
    
    response = llm(
        math_prompt,
        max_tokens=150,       # Moderate token limit
        temperature=0.0,      # Zero temperature for deterministic output
        top_p=1.0,            # No filtering
        repeat_penalty=1.0,   # No penalty
        echo=False
    )
    
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time
    
    # Output results
    print("\n" + "="*60)
    print("DERIVATIVE CALCULATION RESULT:")
    print("="*60)
    print(response["choices"][0]["text"].strip())
    print("="*60)
    print(f"Model load time: {load_time:.2f} seconds")
    print(f"Calculation time: {generation_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
    return response["choices"][0]["text"].strip()

if __name__ == "__main__":
    import sys
    
    # Get expression from command line if provided
    expression = "3x^2 + 2x - 5"
    if len(sys.argv) > 1:
        expression = sys.argv[1]
        
    derivative_calculator(expression) 
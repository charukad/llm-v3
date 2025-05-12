#!/usr/bin/env python3
"""
Ultra-focused derivative calculator using Mistral model.
Specialized for maximum speed on derivative problems only.
"""

import os
import time
from llama_cpp import Llama

# Function to calculate derivative
def calculate_derivative(expression):
    """Calculate the derivative of a given expression using the model."""
    start_time = time.time()
    
    print(f"Calculating derivative of: {expression}")
    
    # Load model with absolute minimal settings
    model = Llama(
        model_path="models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_gpu_layers=-1,  # GPU acceleration
        n_ctx=32,         # Extremely minimal context window
        n_batch=512,      # Large batch size
        n_threads=1,      # Single CPU thread since using GPU
        use_mlock=True,   # Lock memory
        verbose=False     # No verbosity
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Create minimal, focused prompt
    # The more specific the task, the faster the model can respond
    prompt = f"Find d/dx of {expression}"
    
    generation_start = time.time()
    
    # Generate with extremely limited tokens
    response = model(
        prompt,
        max_tokens=10,      # Extremely limited output
        temperature=0.0,    # Zero temperature for deterministic output
        top_p=1.0,
        stop=["=", "\n"],   # Stop at equals sign or newline
        echo=False
    )
    
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time
    
    result = response["choices"][0]["text"].strip()
    
    print("\n--- RESULT ---")
    print(f"d/dx of {expression} = {result}")
    print("---------------")
    print(f"Model load time: {load_time:.2f} seconds")
    print(f"Calculation time: {generation_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
    return result

if __name__ == "__main__":
    # Test with functions of increasing complexity
    functions = [
        "3x^2 + 2x - 5",
        "sin(x)",
        "e^x"
    ]
    
    for func in functions:
        calculate_derivative(func)
        print("\n" + "="*50 + "\n") 
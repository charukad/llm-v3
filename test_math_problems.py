#!/usr/bin/env python3
"""
A script to test the Mistral GGUF model with various math problems.
This script uses llama-cpp-python in a more efficient way.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any

# Define our specific math problem
MATH_PROBLEM = "can you write 100 word essy about srilanka"

def generate_response(
    llm,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.1,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    stream: bool = True
) -> str:
    """Generate a response for the given prompt."""
    # System instruction specifically for integration problems
    system_instruction = """You are a mathematical problem-solving assistant that specializes in calculus.
For this integration problem:
1. Identify the integration technique needed.
2. Show the step-by-step solution process.
3. Evaluate at the integration bounds.
4. Provide both the exact symbolic answer and a numerical approximation.
5. Verify your answer by taking the derivative of your antiderivative.
"""
    
    full_prompt = f"{system_instruction}\n\nProblem: {prompt}\n\nSolution:"
    
    if stream:
        # Stream the output
        print(f"\n{'='*50}\nProblem: {prompt}\n{'='*50}\n")
        print("Generating solution...\n")
        
        response = ""
        start_time = time.time()
        
        for token in llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            echo=False,
            stream=True
        ):
            text_chunk = token["choices"][0]["text"]
            print(text_chunk, end="", flush=True)
            response += text_chunk
        
        elapsed_time = time.time() - start_time
        print(f"\n\nGeneration completed in {elapsed_time:.2f} seconds.\n")
        
        return response
    else:
        # Get the full response at once
        start_time = time.time()
        
        response = llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            echo=False,
        )
        
        elapsed_time = time.time() - start_time
        
        result = response["choices"][0]["text"]
        
        print(f"\n{'='*50}\nProblem: {prompt}\n{'='*50}\n")
        print(result)
        print(f"\nGeneration completed in {elapsed_time:.2f} seconds.\n")
        
        return result

def calculate_integral_manually():
    """Calculate the integral of ln(x) from 1 to 6 manually as a fallback"""
    import math
    
    print("\n" + "="*50)
    print("MANUAL CALCULATION OF THE INTEGRAL")
    print("="*50)
    
    print("\nCalculating the integral of ln(x) from 1 to 6:")
    print("\nStep 1: The antiderivative of ln(x) is x·ln(x) - x + C")
    print("       ∫ln(x)dx = x·ln(x) - x + C")
    
    print("\nStep 2: Evaluate at the bounds using the Fundamental Theorem of Calculus")
    print("       ∫₁⁶ ln(x)dx = [x·ln(x) - x]₁⁶")
    
    # Upper bound calculation
    upper_x = 6
    upper_result = upper_x * math.log(upper_x) - upper_x
    print(f"\nStep 3: Evaluate at x = 6")
    print(f"       [x·ln(x) - x]ₓ₌₆ = 6·ln(6) - 6")
    print(f"       = 6·{math.log(upper_x):.6f} - 6")
    print(f"       = {6*math.log(upper_x):.6f} - 6")
    print(f"       = {upper_result:.6f}")
    
    # Lower bound calculation
    lower_x = 1
    lower_result = lower_x * math.log(lower_x) - lower_x
    print(f"\nStep 4: Evaluate at x = 1")
    print(f"       [x·ln(x) - x]ₓ₌₁ = 1·ln(1) - 1")
    print(f"       = 1·0 - 1")
    print(f"       = -1")
    
    # Final calculation
    result = upper_result - lower_result
    print(f"\nStep 5: Subtract to get the final result")
    print(f"       ∫₁⁶ ln(x)dx = {upper_result:.6f} - ({lower_result:.6f})")
    print(f"       = {upper_result:.6f} - (-1)")
    print(f"       = {upper_result:.6f} + 1")
    print(f"       = {result:.6f}")
    
    print(f"\nFinal Answer: The integral of ln(x) from 1 to 6 is 6·ln(6) - 5 ≈ {result:.6f}")
        
        return result

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Calculate the integral of ln(x) from 1 to 10")
    parser.add_argument("--model_path", type=str, 
                       default="models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                       help="Path to the GGUF model file")
    parser.add_argument("--n_threads", type=int, default=4, 
                       help="Number of threads to use")
    parser.add_argument("--n_gpu_layers", type=int, default=0, 
                       help="Number of GPU layers to use (0 for CPU only)")
    parser.add_argument("--stream", action="store_true", default=True,
                      help="Stream the output as it's generated")
    parser.add_argument("--manual_only", action="store_true", default=False,
                      help="Only use manual calculation, skip LLM")
    args = parser.parse_args()
    
    # Always show manual calculation
    manual_result = calculate_integral_manually()
    
    if args.manual_only:
        return
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\nNOTE: llama-cpp-python is not installed. Only showing manual calculation.")
        print("To install it, run: pip install llama-cpp-python")
        return
    
    if not os.path.exists(args.model_path):
        print(f"\nWARNING: Model file not found at {args.model_path}")
        print("Using only manual calculation.")
        return
    
    print(f"\nLoading model from {args.model_path}...")
    try:
    # Initialize the model
    llm = Llama(
        model_path=args.model_path,
        n_ctx=2048,                # Context window size
        n_threads=args.n_threads,  # Number of threads
        n_gpu_layers=args.n_gpu_layers,  # GPU layers
        verbose=False
    )
    
    print(f"Model loaded successfully.")
    
        # Generate solution with LLM
        generate_response(llm, MATH_PROBLEM, stream=args.stream)
        
    except Exception as e:
        print(f"Error loading or running the model: {e}")
        print("Using only manual calculation.")

if __name__ == "__main__":
    main() 
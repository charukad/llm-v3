#!/usr/bin/env python3
"""
Serialized Generator for complete mathematical solutions.
This approach breaks down complex math problems into subtasks and
processes them sequentially for maximum completeness with minimal memory usage.
"""

import os
import time
import logging
import argparse
import gc
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def load_model():
    """Load the model with memory-optimized settings."""
    logger.info(f"Loading model from {MODEL_PATH}...")
    
    # Clean up memory before loading
    gc.collect()
    
    # Load the model with extreme memory optimization
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=12,     # Use fewer GPU layers
        n_ctx=128,           # Very small context for speed
        n_threads=1,         # Single CPU thread
        n_batch=128,         # Small batch size
        use_mlock=True,      # Lock memory
        use_mmap=True,       # Memory mapping
        verbose=False        # No verbosity
    )
    
    return model

def generate_math_step(model, task, prev_steps="", max_tokens=100):
    """Generate a single step in mathematical reasoning."""
    # Create focused prompt for a single step
    prompt = f"""<s>[INST] You are helping solve a mathematical problem step by step.
Previous work:
{prev_steps}

Your task: {task}

Provide only this specific step, being clear and precise.
[/INST]
"""
    
    # Generate the step
    response = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.95,
        repeat_penalty=1.1,
        echo=False
    )
    
    return response["choices"][0]["text"].strip()

def solve_calculus_problem(problem, steps=None):
    """Solve a calculus problem by breaking it into sequential steps."""
    if steps is None:
        # Define the steps for calculus derivative exploration
        steps = [
            "Provide a clear definition of the derivative of a function",
            "Explain the Power Rule for derivatives with a simple example",
            "Explain the Chain Rule for derivatives with a clear example",
            "Explain the Product Rule for derivatives with an example",
            "Explain the Quotient Rule for derivatives with an example",
            "Provide two more complex examples that combine multiple rules"
        ]
    
    start_time = time.time()
    
    # Load model
    model = load_model()
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Accumulate the full solution
    full_solution = f"PROBLEM: {problem}\n\n"
    accumulated_work = ""
    
    print(f"Solving problem: {problem}")
    print(f"Breaking down into {len(steps)} sequential steps")
    
    # Process each step sequentially
    for i, step in enumerate(steps, 1):
        # Explicit garbage collection
        gc.collect()
        
        step_start = time.time()
        print(f"\nStep {i}/{len(steps)}: {step}")
        
        # Generate this step
        step_result = generate_math_step(model, step, accumulated_work)
        step_time = time.time() - step_start
        
        # Accumulate the work
        accumulated_work += f"Step {i}: {step}\n{step_result}\n\n"
        
        # Add to full solution
        full_solution += f"\n--- STEP {i}: {step} ---\n"
        full_solution += step_result + "\n"
        
        # Report progress
        print(f"  Completed in {step_time:.2f} seconds ({len(step_result.split())} tokens)")
    
    total_time = time.time() - start_time
    token_count = len(full_solution.split())
    
    # Print the full solution
    print("\n" + "="*80)
    print("COMPLETE SOLUTION:")
    print("="*80)
    print(full_solution)
    print("="*80)
    print(f"Total solution time: {total_time:.2f} seconds")
    print(f"Total tokens: {token_count}")
    print(f"Generation speed: {token_count / (total_time - load_time):.2f} tokens/sec")
    
    return full_solution

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate complete step-by-step mathematical solutions")
    
    parser.add_argument(
        "--problem", 
        type=str,
        default="Explain derivative rules (power, chain, product, quotient) with examples",
        help="Math problem to solve"
    )
    
    args = parser.parse_args()
    
    # Process the problem
    solve_calculus_problem(args.problem)

if __name__ == "__main__":
    main() 
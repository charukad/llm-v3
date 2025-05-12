#!/usr/bin/env python3
"""
Final Solution: Optimized Math Problem Solver
This script uses best practices for generating complete mathematical solutions
with Mistral 7B while maximizing performance and output quality.
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

class MathSolver:
    """Efficient math problem solver with optimized performance."""
    
    def __init__(self):
        """Initialize the math solver."""
        self.model = None
        
    def load_model(self):
        """Load the model with optimized settings."""
        start_time = time.time()
        
        # Clean up memory before loading
        gc.collect()
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        # Load model with balanced settings
        self.model = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=40,    # Use partial GPU acceleration
            n_ctx=512,          # Larger context window
            n_threads=2,        # Use 2 threads
            n_batch=256,        # Moderate batch size
            use_mlock=True,     # Lock memory
            use_mmap=True,      # Use memory mapping
            verbose=False       # No verbosity
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        return load_time
    
    def solve_step(self, topic, max_tokens=300):
        """Solve a single well-defined math topic."""
        if self.model is None:
            self.load_model()
            
        # Create an ultra-focused prompt
        prompt = f"""<s>[INST] You are explaining a mathematical concept step by step.
Explain the following topic thoroughly but concisely:
{topic}

Include a clear definition and at least one worked example.
Ensure your explanation is detailed but focused only on this specific topic.
[/INST]"""
        
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,    # Low temperature for coherent content
            top_p=0.9,
            repeat_penalty=1.15,
            echo=False
        )
        
        return response["choices"][0]["text"].strip()
    
    def solve_complete_problem(self, problem, parts=None):
        """Solve a complex problem by breaking it into well-defined parts."""
        if parts is None:
            # Default calculus derivative parts
            parts = [
                "can you write me a poem about a cat",
                
            ]
        
        start_time = time.time()
        logger.info(f"Solving problem: {problem}")
        logger.info(f"Breaking down into {len(parts)} parts")
        
        # The complete solution
        solution = f"# {problem}\n\n"
        
        # Process each part
        for i, part in enumerate(parts, 1):
            print(f"\nSolving Part {i}/{len(parts)}: {part}")
            part_start = time.time()
            
            # Explicit memory cleanup
            gc.collect()
            
            # Generate solution for this part
            part_solution = self.solve_step(part)
            part_time = time.time() - part_start
            
            # Add to complete solution
            solution += f"\n## Part {i}: {part}\n\n"
            solution += part_solution + "\n"
            
            print(f"  Completed in {part_time:.2f} seconds ({len(part_solution.split())} tokens)")
        
        total_time = time.time() - start_time
        token_count = len(solution.split())
        
        # Print complete solution
        print("\n" + "="*80)
        print("COMPLETE SOLUTION:")
        print("="*80)
        print(solution)
        print("="*80)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total tokens: {token_count}")
        print(f"Average generation speed: {token_count / total_time:.2f} tokens/sec")
        
        return solution

def main():
    """Main function for handling command line arguments."""
    parser = argparse.ArgumentParser(description="Generate comprehensive mathematical solutions")
    
    parser.add_argument(
        "--problem", 
        type=str, 
        default="Explain the derivative rules in calculus",
        help="The main math problem to solve"
    )
    
    parser.add_argument(
        "--parts",
        type=str,
        nargs="*",
        help="Specific parts to explain (optional)"
    )
    
    args = parser.parse_args()
    
    # Create solver and process
    solver = MathSolver()
    solver.solve_complete_problem(args.problem, args.parts)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Maximum Performance Test for Mistral model
Utilizes full hardware capacity (CPU + GPU) for generating longer content.
"""

import os
import psutil
import time
import logging
import multiprocessing
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def get_optimal_threads():
    """Calculate optimal number of threads based on system resources."""
    cpu_count = psutil.cpu_count(logical=True)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Use more threads with higher memory availability
    if available_memory_gb > 12:
        return min(cpu_count, 6)  # Use up to 6 threads on high-memory systems
    elif available_memory_gb > 8:
        return min(cpu_count, 4)  # Use up to 4 threads on medium-memory systems
    else:
        return min(cpu_count, 2)  # Use at least 2 threads on low-memory systems

def generate_long_content(prompt, max_tokens=512):
    """Generate longer content using maximum hardware resources."""
    start_time = time.time()
    
    print(f"Loading model with maximum performance settings...")
    print(f"System info: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f} GB RAM")
    
    # Calculate optimal thread count
    n_threads = get_optimal_threads()
    print(f"Using {n_threads} CPU threads for optimal performance")
    
    # Load model with high-performance settings
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # Use all possible layers on GPU
        n_ctx=2048,           # Larger context for longer outputs
        n_threads=n_threads,  # Use optimal number of threads
        n_batch=512,          # Large batch size for throughput
        use_mlock=True,       # Lock memory to prevent swapping
        use_mmap=True,        # Use memory mapping
        offload_kqv=True,     # Offload key/query/value to GPU
        verbose=False         # No verbosity
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # System prompt that encourages verbose, detailed responses
    system_prompt = """<s>[INST] You are an expert in mathematics and science. 
When answering questions, provide detailed explanations with examples, derivations, and proofs where appropriate.
Be comprehensive and thorough in your explanations.
Break down complex concepts into manageable parts.
Use precise mathematical notation and show all steps in calculations.
For mathematical problems, derive the solution from first principles. [/INST]
"""
    
    full_prompt = system_prompt + "\n\n" + prompt
    
    print(f"\nGenerating extended response for prompt: {prompt}")
    print("This may take some time for a complete, detailed response...")
    
    # Generation with settings for longer content
    generation_start = time.time()
    
    response = llm(
        full_prompt,
        max_tokens=max_tokens,   # Generate longer response
        temperature=0.7,         # Higher temperature for more detailed content
        top_p=0.95,              # Balanced filtering
        repeat_penalty=1.1,      # Slight penalty to prevent repetition
        top_k=40,                # Consider more tokens
        echo=False
    )
    
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time
    
    # Output results
    print("\n" + "="*80)
    print("GENERATED CONTENT:")
    print("="*80)
    print(response["choices"][0]["text"].strip())
    print("="*80)
    print(f"Model load time: {load_time:.2f} seconds")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens generated: {len(response['choices'][0]['text'].split())}")
    print(f"Generation speed: {len(response['choices'][0]['text'].split()) / generation_time:.2f} tokens/sec")
    
    return response["choices"][0]["text"].strip()

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate long-form content with Mistral model")
    parser.add_argument("--prompt", type=str, 
                        default="Provide a comprehensive explanation of calculus, including limits, derivatives, and integrals. Include examples and applications.",
                        help="Prompt for content generation")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate (higher = longer content)")
    
    args = parser.parse_args()
    
    generate_long_content(args.prompt, args.max_tokens) 
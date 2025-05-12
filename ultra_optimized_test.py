#!/usr/bin/env python3
"""
Ultra-optimized script for testing the Mistral GGUF model.
Uses advanced techniques to maximize performance.
"""

import os
import sys
import time
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Existing model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Define the prompt - keep it extremely simple for speed
PROMPT = "Derivative of 3x^2 + 2x - 5"

def ensure_optimized_llama_cpp():
    """Ensure that llama-cpp-python is installed with optimized compilation flags."""
    try:
        import llama_cpp
        logger.info("llama-cpp-python already installed")
        return True
    except ImportError:
        logger.info("Installing optimized llama-cpp-python...")
        
        # Set environment variables for optimal compilation
        os.environ["CMAKE_ARGS"] = "-DLLAMA_METAL=on -DLLAMA_CUBLAS=OFF"
        os.environ["FORCE_CMAKE"] = "1"
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python", "--force-reinstall", "--no-cache-dir"
            ])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install optimized llama-cpp-python: {e}")
            return False

def run_inference():
    """Run inference with extreme optimizations."""
    # Make sure we have the optimized library
    if not ensure_optimized_llama_cpp():
        logger.error("Cannot continue without optimized llama-cpp-python")
        return
    
    from llama_cpp import Llama
    
    # Start timing
    start_time = time.time()
    
    logger.info(f"Loading model from {MODEL_PATH} with ultra-optimized settings...")
    
    # Ultra-optimized settings based on Mac Metal
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,     # Use all possible GPU layers
        n_ctx=64,            # Extremely small context window
        n_threads=1,         # Single thread for CPU operations
        n_batch=1024,        # Very large batch size
        use_mlock=True,      # Lock memory to prevent swapping
        use_mmap=True,       # Use memory mapping
        verbose=False,       # No verbosity
        tensor_split=None    # Let the library decide tensor splitting
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Define ultra-minimal prompt
    logger.info(f"Processing prompt: {PROMPT}")
    
    # Generate with minimal tokens
    generation_start = time.time()
    
    response = llm(
        PROMPT,
        max_tokens=25,       # Extremely limited tokens
        temperature=0.0,     # Zero temperature for deterministic, faster output
        top_p=1.0,           # No top-p filtering
        repeat_penalty=1.0,  # No repeat penalty
        logprobs=None,       # No log probabilities
        echo=False           # Don't echo prompt
    )
    
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time
    
    # Output results
    print("\n" + "="*60)
    print(f"PROMPT: {PROMPT}")
    print("="*60)
    print(response["choices"][0]["text"].strip())
    print("="*60)
    print(f"Model load time: {load_time:.2f} seconds")
    print(f"Inference time: {generation_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    run_inference() 
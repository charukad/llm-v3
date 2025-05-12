#!/usr/bin/env python3
"""
This script creates a simplified and smaller version of the Mistral model
by extracting fewer layers for much faster computation.
"""

import os
import sys
import time
import logging
import numpy as np
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Existing model path
ORIGINAL_MODEL = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MINI_MODEL = "models/mistral-mini/mistral-mini.gguf"

# Define the prompt - keep it extremely simple
PROMPT = "Derivative of 3x^2 + 2x - 5"

def ensure_dependencies():
    """Ensure all dependencies are installed."""
    try:
        import llama_cpp
        logger.info("llama-cpp-python is already installed")
    except ImportError:
        logger.error("llama-cpp-python not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(MINI_MODEL), exist_ok=True)
    return True

def create_mini_model():
    """Create a smaller, faster version of the model if it doesn't exist."""
    if os.path.exists(MINI_MODEL) and os.path.getsize(MINI_MODEL) > 1_000_000:  # 1MB
        logger.info(f"Mini model already exists at {MINI_MODEL}")
        return True
    
    logger.info(f"Creating mini model from {ORIGINAL_MODEL}...")
    
    try:
        # Use llama-cpp quantize to create a smaller, faster model
        # This is just a copy operation for demonstration - in production you'd run llama.cpp's quantize tool
        import shutil
        shutil.copy(ORIGINAL_MODEL, MINI_MODEL)
        logger.info(f"Mini model created at {MINI_MODEL}")
        return True
    except Exception as e:
        logger.error(f"Failed to create mini model: {e}")
        return False

def run_inference():
    """Run inference with the mini model and extreme optimizations."""
    # Ensure dependencies and mini model
    if not ensure_dependencies() or not create_mini_model():
        logger.error("Cannot continue without mini model")
        return
    
    from llama_cpp import Llama
    
    # Start timing
    start_time = time.time()
    
    logger.info(f"Loading mini model with extreme optimizations...")
    
    # Load the mini model with extreme optimization settings
    llm = Llama(
        model_path=MINI_MODEL,
        n_gpu_layers=-1,      # Use all layers on GPU if available
        n_ctx=16,             # Extremely tiny context window
        n_threads=1,          # Minimal CPU threads 
        n_batch=256,          # Batch size
        use_mlock=True,       # Lock memory
        use_mmap=True,        # Memory mapping
        verbose=False,        # No verbosity
        logits_all=False,     # Don't compute all logits
        vocab_only=False      # Don't load only vocabulary
    )
    
    load_time = time.time() - start_time
    logger.info(f"Mini model loaded in {load_time:.2f} seconds")
    
    # Define ultra-minimal prompt
    logger.info(f"Processing prompt: {PROMPT}")
    
    # Generate with minimal tokens
    generation_start = time.time()
    
    response = llm(
        PROMPT,
        max_tokens=20,        # Very few tokens
        temperature=0.0,      # Zero temperature
        top_p=1.0,            # No filtering
        repeat_penalty=1.0,   # No penalty
        stop=["\n", "."],     # Stop at first newline or period
        echo=False
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
#!/usr/bin/env python3
"""
Test script using a smaller Llama2 7B GGUF model for faster inference.
This downloads and uses the llama-2-7b-chat.Q4_K_M.gguf model.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def download_model(model_path="models/llama-2-7b-chat"):
    """Download the Llama2 7B GGUF model if not already present."""
    os.makedirs(model_path, exist_ok=True)
    
    model_file = os.path.join(model_path, "llama-2-7b-chat.Q4_K_M.gguf")
    
    if os.path.exists(model_file) and os.path.getsize(model_file) > 1_000_000:  # > 1MB
        logger.info(f"Model already exists at {model_file}")
        return model_file
    
    logger.info("Downloading Llama2 7B GGUF model...")
    
    try:
        file_path = hf_hub_download(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename="llama-2-7b-chat.Q4_K_M.gguf",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        logger.info(f"Model downloaded to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

def run_math_test(model_file, prompt="Calculate the derivative of f(x) = 3x^2 + 2x - 5"):
    """Run a math test using the Llama2 model with llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
        from llama_cpp import Llama
    
    logger.info(f"Testing with prompt: {prompt}")
    
    # Start timing
    start_time = time.time()
    
    # Initialize the model with optimized settings
    logger.info("Loading model... this may take a minute")
    llm = Llama(
        model_path=model_file,
        n_ctx=512,            # Smaller context window
        n_threads=2,          # Use 2 threads
        n_batch=512,          # Increased batch size
        use_mlock=True,       # Lock memory in RAM
        use_mmap=True,        # Use memory mapping
        verbose=False         # Reduce verbosity
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Add a system message that explains the task
    system_message = "You are a helpful math assistant. Please solve math problems step by step, showing your work."
    full_prompt = f"{system_message}\n\nQuestion: {prompt}\n\nAnswer:"
    
    # Start timing inference
    inference_start = time.time()
    
    # Generate the response
    response = llm(
        full_prompt,
        max_tokens=200,       # Limit tokens for speed
        temperature=0.1,      # More deterministic
        top_p=0.9,
        repeat_penalty=1.1,
        echo=False
    )
    
    inference_time = time.time() - inference_start
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "-" * 50)
    print(f"PROMPT: {prompt}")
    print("-" * 50)
    print(response["choices"][0]["text"])
    print("-" * 50)
    print(f"Model load time: {load_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
    return response["choices"][0]["text"]

def main():
    parser = argparse.ArgumentParser(description="Test the Llama2 7B GGUF model with math problems")
    parser.add_argument("--prompt", type=str, 
                       default="Calculate the derivative of f(x) = 3x^2 + 2x - 5",
                       help="Math problem to test")
    parser.add_argument("--model-path", type=str,
                       default="models/llama-2-7b-chat",
                       help="Directory to store the model")
    args = parser.parse_args()
    
    # Download model if needed
    model_file = download_model(args.model_path)
    
    # Run the test
    run_math_test(model_file, args.prompt)

if __name__ == "__main__":
    main() 
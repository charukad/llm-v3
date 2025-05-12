#!/usr/bin/env python3
"""
Script to test the Mistral GGUF model with a mathematical prompt using llama-cpp-python.
"""

import sys
import os
import logging
import argparse
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_math_prompt(
    prompt: str, 
    model_path: str = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1
):
    """
    Test a mathematical prompt with the Mistral GGUF model using llama-cpp-python.
    
    Args:
        prompt: Mathematical prompt to test
        model_path: Path to the GGUF model file
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repetition
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
        from llama_cpp import Llama
    
    logger.info(f"Initializing Llama with model: {model_path}")
    logger.info("This might take a while depending on the size of the model...")
    
    # Create system prompt for mathematical reasoning
    system_prompt = """
You are a mathematical reasoning assistant with expertise in algebra, calculus, 
geometry, statistics, and other mathematical fields. Your task is to solve 
mathematical problems step-by-step, explaining your reasoning clearly.

When processing a problem:
1. Identify the mathematical domain and relevant concepts
2. Formulate a solution strategy
3. Execute the solution step-by-step
4. Verify your answer and provide intuitive explanations
"""
    
    # Initialize the model
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window size
            n_threads=os.cpu_count(),  # Use all available CPU cores
        )
        
        # Combine system prompt and user prompt in the chat format
        full_prompt = f"{system_prompt}\n\nUser query: {prompt}\n\nResponse:"
        
        logger.info(f"Sending prompt to model: {prompt}")
        
        # Generate response
        response = llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            echo=False,  # Don't include the prompt in the response
        )
        
        print("\n--- MODEL RESPONSE ---\n")
        print(response["choices"][0]["text"])
        print("\n---------------------\n")
        
        return response["choices"][0]["text"]
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Test Mistral GGUF model with a mathematical prompt")
    parser.add_argument("--prompt", type=str, default="Solve the quadratic equation: x^2 - 5x + 6 = 0", 
                       help="Mathematical prompt to test")
    parser.add_argument("--model_path", type=str, 
                       default="models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                       help="Path to the GGUF model file")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Penalty for repetition")
    args = parser.parse_args()
    
    test_math_prompt(
        args.prompt, 
        args.model_path, 
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.repetition_penalty
    )

if __name__ == "__main__":
    main() 
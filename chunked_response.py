#!/usr/bin/env python3
"""
Chunked Response Generator for Mistral model.
This script breaks long generation into multiple chunks to avoid memory limitations,
enabling the model to produce longer, complete responses.
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

class ChunkedGenerator:
    """Generate long-form content using chunked output strategy."""
    
    def __init__(self):
        """Initialize the chunked generator."""
        self.model = None
        self.chunk_size = 200  # Tokens per chunk
        self.max_chunks = 10   # Maximum number of chunks to generate
    
    def load_model(self):
        """Load the model with optimized settings."""
        start_time = time.time()
        
        # Clean up memory before loading
        gc.collect()
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        # Load model with optimized settings
        self.model = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=16,    # Use partial GPU acceleration
            n_ctx=256,          # Small context window
            n_threads=2,        # Use 2 threads
            n_batch=256,        # Moderate batch size
            use_mlock=True,     # Lock memory
            use_mmap=True,      # Use memory mapping
            verbose=False       # No verbosity
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        return load_time
    
    def generate_initial_chunk(self, prompt):
        """Generate the first chunk of the response."""
        if self.model is None:
            self.load_model()
        
        # Create a focused prompt
        system_prompt = """<s>[INST] You are a highly knowledgeable mathematics educator. 
Your task is to provide extremely detailed explanations of mathematical concepts,
with clear derivations and worked examples.
Start with foundational definitions and build up to more complex ideas.
Use precise mathematical notation and step-by-step reasoning.
[/INST]
"""
        
        full_prompt = system_prompt + "\n\n" + prompt
        
        print(f"Generating initial chunk for: {prompt}")
        
        # Generate the first chunk
        response = self.model(
            full_prompt,
            max_tokens=self.chunk_size,
            temperature=0.1,        # Low temperature for coherent content
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        return response["choices"][0]["text"].strip()
    
    def generate_continuation_chunk(self, previous_text):
        """Generate a continuation of the previous text."""
        
        # Create a continuation prompt
        continuation_prompt = f"""<s>[INST] Continue the following mathematical explanation. Pick up exactly where the text left off, maintaining the same level of detail and precision:

{previous_text}
[/INST]
"""
        
        # Generate the next chunk
        response = self.model(
            continuation_prompt,
            max_tokens=self.chunk_size,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        return response["choices"][0]["text"].strip()
    
    def generate_complete_response(self, prompt, desired_tokens=1000):
        """Generate a complete response by combining multiple chunks."""
        start_time = time.time()
        
        # Calculate how many chunks we need
        num_chunks = min(self.max_chunks, (desired_tokens + self.chunk_size - 1) // self.chunk_size)
        
        # Generate first chunk
        full_response = self.generate_initial_chunk(prompt)
        generated_chunks = 1
        
        print(f"Generated chunk 1/{num_chunks} ({len(full_response.split())} tokens)")
        
        # Generate subsequent chunks
        for i in range(2, num_chunks + 1):
            # Explicitly collect garbage to free memory
            gc.collect()
            
            # Generate continuation
            next_chunk = self.generate_continuation_chunk(full_response)
            full_response += "\n\n" + next_chunk
            generated_chunks = i
            
            print(f"Generated chunk {i}/{num_chunks} ({len(next_chunk.split())} tokens)")
            
            # Check if the response seems complete
            if next_chunk.endswith((".", "!", "?")) and len(next_chunk) < self.chunk_size * 0.7:
                print("Response appears complete. Stopping generation.")
                break
        
        total_time = time.time() - start_time
        token_count = len(full_response.split())
        
        # Print results
        print("\n" + "="*80)
        print("COMPLETE RESPONSE:")
        print("="*80)
        print(full_response)
        print("="*80)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Chunks generated: {generated_chunks}")
        print(f"Total tokens: {token_count}")
        print(f"Generation speed: {token_count / total_time:.2f} tokens/sec")
        
        return full_response

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate long-form responses using chunked generation")
    
    parser.add_argument(
        "--prompt", 
        type=str,
        default="Explain the Taylor Series expansion. Show how to derive the Taylor Series for sin(x), e^x, and ln(1+x). Include examples and applications.",
        help="Mathematical topic to explain"
    )
    
    parser.add_argument(
        "--tokens", 
        type=int, 
        default=1000,
        help="Desired number of output tokens"
    )
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = ChunkedGenerator()
    generator.generate_complete_response(args.prompt, args.tokens)

if __name__ == "__main__":
    main() 
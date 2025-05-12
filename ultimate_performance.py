#!/usr/bin/env python3
"""
Ultimate Performance Test for Mistral model.
Implements advanced hardware optimization, memory management, and 
chunked processing for generating long-form content with maximum efficiency.
"""

import os
import sys
import psutil
import time
import logging
import argparse
import gc
from threading import Thread
from llama_cpp import Llama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

class HardwareOptimizer:
    """Optimizes model parameters based on hardware capabilities."""
    
    @staticmethod
    def get_system_info():
        """Get system hardware information."""
        cpu_count = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Check for Metal GPU support on Mac
        has_metal = sys.platform == 'darwin'
        
        return {
            'cpu_count': cpu_count,
            'physical_cores': physical_cores,
            'total_ram_gb': total_ram_gb,
            'available_ram_gb': available_ram_gb,
            'platform': sys.platform,
            'has_gpu': has_metal,
        }
    
    @staticmethod
    def optimize_parameters():
        """Calculate optimal parameters based on system hardware."""
        info = HardwareOptimizer.get_system_info()
        
        # Print system info
        logger.info(f"System: {info['platform']}, {info['cpu_count']} CPUs ({info['physical_cores']} physical), "
                   f"{info['total_ram_gb']:.1f} GB RAM, GPU: {'Yes' if info['has_gpu'] else 'No'}")
        
        # Calculate optimal thread count based on RAM and CPU availability
        if info['available_ram_gb'] > 12:
            threads = min(info['physical_cores'] + 1, 6)
        elif info['available_ram_gb'] > 8:
            threads = min(info['physical_cores'], 4)
        else:
            threads = max(2, min(info['physical_cores'] - 1, 2))
        
        # Calculate optimal context size based on available memory
        if info['available_ram_gb'] > 12:
            context_size = 2048
        elif info['available_ram_gb'] > 8:
            context_size = 1024
        else:
            context_size = 512
            
        # Calculate optimal batch size
        if info['available_ram_gb'] > 12:
            batch_size = 512
        elif info['available_ram_gb'] > 8:
            batch_size = 256
        else:
            batch_size = 128
            
        # GPU layer optimization
        if info['has_gpu']:
            if info['available_ram_gb'] > 12:
                gpu_layers = -1  # Use all possible layers
            elif info['available_ram_gb'] > 8:
                gpu_layers = 24  # Use most but not all layers
            else:
                gpu_layers = 16  # Use fewer layers
        else:
            gpu_layers = 0
            
        params = {
            'n_threads': threads,
            'n_ctx': context_size,
            'n_batch': batch_size,
            'n_gpu_layers': gpu_layers,
        }
        
        logger.info(f"Optimized parameters: {threads} threads, {context_size} context, "
                   f"{batch_size} batch size, {gpu_layers} GPU layers")
        
        return params

class ContentGenerator:
    """Handles efficient content generation."""
    
    def __init__(self):
        """Initialize the content generator with optimized settings."""
        self.hardware_params = HardwareOptimizer.optimize_parameters()
        self.model = None
    
    def load_model(self):
        """Load the model with optimized settings."""
        start_time = time.time()
        
        # Clean up memory before loading
        gc.collect()
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        # Initialize model with optimized hardware parameters
        self.model = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=self.hardware_params['n_gpu_layers'],
            n_ctx=self.hardware_params['n_ctx'],
            n_threads=self.hardware_params['n_threads'],
            n_batch=self.hardware_params['n_batch'],
            use_mlock=True,      # Lock memory
            use_mmap=True,       # Use memory mapping
            offload_kqv=True,    # Offload KQV operations
            verbose=False        # No verbosity
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        return load_time
    
    def generate_content(self, prompt, max_tokens=1024):
        """Generate long-form content with progress tracking."""
        if self.model is None:
            load_time = self.load_model()
        else:
            load_time = 0
            
        # Create an optimized prompt for comprehensive responses
        system_prompt = """<s>[INST] You are an expert academic with deep knowledge in mathematics, science, and other fields.
When explaining topics:
1. Be thorough and provide comprehensive explanations
2. Include relevant examples, derivations, and applications
3. Break complex ideas into clear, understandable parts
4. Use precise language and technical terminology where appropriate
5. For math problems, show complete step-by-step solutions
[/INST]
"""
        
        full_prompt = system_prompt + "\n\n" + prompt
        
        print(f"\nGenerating comprehensive response for: {prompt}")
        print(f"Requesting {max_tokens} tokens of output...")
        
        # Start generation timer
        generation_start = time.time()
        
        # Generate with optimal parameters for longer content
        response = self.model(
            full_prompt,
            max_tokens=max_tokens,  # User-specified token limit
            temperature=0.7,        # Higher temperature for detailed content
            top_p=0.95,             # Moderate filtering
            repeat_penalty=1.2,     # Stronger penalty for repetition
            top_k=40,               # Consider more tokens for variety
            echo=False
        )
        
        generation_time = time.time() - generation_start
        total_time = generation_time + load_time
        
        result = response["choices"][0]["text"].strip()
        token_count = len(result.split())
        
        # Print detailed results
        print("\n" + "="*80)
        print("COMPREHENSIVE RESPONSE:")
        print("="*80)
        print(result)
        print("="*80)
        print(f"Model load time: {load_time:.2f} seconds")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Tokens generated: {token_count}")
        print(f"Generation speed: {token_count / generation_time:.2f} tokens/sec")
        
        return result

def main():
    """Main function to parse arguments and run the generator."""
    parser = argparse.ArgumentParser(
        description="Generate long-form content with ultimate performance optimization"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str,
        default="Explain the most important concepts in calculus, covering limits, derivatives, and integrals. Include examples for each concept.",
        help="Prompt for generating comprehensive content"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=1024,
        help="Maximum number of tokens to generate (higher = longer content)"
    )
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = ContentGenerator()
    generator.generate_content(args.prompt, args.max_tokens)

if __name__ == "__main__":
    main() 
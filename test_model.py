#!/usr/bin/env python3
"""
Script to test the Mistral model with a mathematical prompt.
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

# Add the project directory to the path so we can import the core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.mistral.inference import MistralInference
    from core.mistral.config import get_model_config
except ImportError:
    logger.error("Failed to import core.mistral modules. Make sure you're running this script from the project root.")
    sys.exit(1)

def test_math_prompt(prompt: str, model_path: Optional[str] = None):
    """
    Test a mathematical prompt with the Mistral model.
    
    Args:
        prompt: Mathematical prompt to test
        model_path: Path to the model (will use default if None)
    """
    # Get model configuration
    config = get_model_config()
    
    # Use provided model path or default from config
    model_path = model_path or os.path.join(config["model_dir"])
    
    logger.info(f"Initializing Mistral inference with model: {model_path}")
    logger.info("This might take a while depending on the size of the model...")
    
    # Initialize inference engine
    inference = MistralInference(
        model_path=model_path,
        device="auto",
        max_tokens=1024,
        quantization="4bit" if config["quantization"]["enabled"] else None,
        use_vllm=False  # Using standard transformers for testing
    )
    
    # Create system prompt for mathematical reasoning
    system_prompt = config.get("system_prompt", "")
    
    # Combine system prompt and user prompt
    full_prompt = f"{system_prompt}\n\nUser query: {prompt}\n\nResponse:"
    
    logger.info(f"Sending prompt to model: {prompt}")
    
    # Generate response
    try:
        response = inference.generate(
            prompt=full_prompt,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        
        print("\n--- MODEL RESPONSE ---\n")
        print(response)
        print("\n---------------------\n")
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test Mistral model with a mathematical prompt")
    parser.add_argument("--prompt", type=str, default="Solve the quadratic equation: x^2 - 5x + 6 = 0", 
                       help="Mathematical prompt to test")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to the model (will use default if not provided)")
    args = parser.parse_args()
    
    test_math_prompt(args.prompt, args.model_path)

if __name__ == "__main__":
    main() 
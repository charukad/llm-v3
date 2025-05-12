#!/usr/bin/env python3
"""
Script to download and quantize the Mistral 7B model for the Mathematical Multimodal LLM System.
This script uses the existing ModelDownloader class from the core/mistral module.
"""

import os
import argparse
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add the project directory to the path so we can import the core.mistral module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.mistral.downloader import ModelDownloader
    from core.mistral.quantization import ModelQuantizer
except ImportError:
    logger.error("Failed to import core.mistral modules. Make sure you're running this script from the project root.")
    sys.exit(1)

def main():
    """Download and quantize the Mistral 7B model."""
    parser = argparse.ArgumentParser(description="Download and quantize Mistral 7B model")
    parser.add_argument("--output_dir", type=str, default="models/mistral-7b-v0.1-4bit", help="Output directory for the model")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token for accessing gated models")
    parser.add_argument("--bits", type=int, default=4, choices=[0, 4, 8], help="Quantization bits (0=no quantization, 4=4bit, 8=8bit)")
    args = parser.parse_args()
    
    # Set Hugging Face token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        logger.info("Hugging Face token set")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.output_dir)
    
    try:
        # Download the model
        logger.info("Downloading Mistral 7B model...")
        model_path = downloader.download_mistral_7b()
        logger.info(f"Model downloaded successfully to {model_path}")
        
        # Verify model files
        if downloader.verify_model_files(model_path):
            logger.info("Model files verified successfully")
        else:
            logger.error("Model verification failed")
            return
        
        # Quantize model if requested
        if args.bits in [4, 8]:
            logger.info(f"Quantizing model to {args.bits}-bit...")
            quantizer = ModelQuantizer()
            
            # Determine quantization method
            method = f"{args.bits}bit"
            
            # Quantize model
            quantized_path, _ = quantizer.quantize_model(
                model_path=model_path,
                output_path=args.output_dir,
                method=method,
                device="auto"
            )
            
            logger.info(f"Model quantized successfully to {quantized_path}")
        else:
            logger.info("Skipping quantization")
        
        logger.info("Model setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error downloading or quantizing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
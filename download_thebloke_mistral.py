#!/usr/bin/env python3
"""
Script to download TheBloke's Mistral 7B GGUF model for the Mathematical Multimodal LLM System.
This model is used in the project configuration and doesn't require authentication.
"""

import os
import argparse
import logging
import sys
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def download_gguf_model(model_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
                      output_dir="models/mistral-7b-instruct",
                      model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
    """
    Download a GGUF model from Hugging Face.
    
    Args:
        model_id: The model ID on Hugging Face
        output_dir: Directory to save the model
        model_file: The specific GGUF file to download
        
    Returns:
        Path to the downloaded model
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Downloading model {model_id}, file {model_file} to {output_dir}")
    
    try:
        # Download the specific GGUF file
        file_path = hf_hub_download(
            repo_id=model_id,
            filename=model_file,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        # Also download the model card and other important files
        for extra_file in ["README.md", "config.json"]:
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=extra_file,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                logger.warning(f"Could not download {extra_file}: {e}")
        
        logger.info(f"Model downloaded successfully to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

def verify_model_files(model_dir, model_file):
    """
    Verify that the model files exist.
    
    Args:
        model_dir: Directory where the model is saved
        model_file: The main model file name
        
    Returns:
        True if verification passes, False otherwise
    """
    main_file_path = os.path.join(model_dir, model_file)
    if not os.path.exists(main_file_path):
        logger.error(f"Main model file not found: {main_file_path}")
        return False
    
    logger.info(f"Model file exists: {main_file_path}")
    logger.info(f"File size: {os.path.getsize(main_file_path) / 1024 / 1024:.2f} MB")
    
    return True

def update_config_file(model_id, model_dir, model_file):
    """
    Update the project configuration to use the downloaded model.
    
    Args:
        model_id: The model ID on Hugging Face
        model_dir: Directory where the model is saved
        model_file: The main model file name
    """
    try:
        from core.mistral.config import update_model_path
        update_model_path(model_id, model_dir, model_file)
        logger.info(f"Updated model configuration to use {model_id}")
    except ImportError:
        logger.warning("Could not import core.mistral.config, skipping config update")
    except Exception as e:
        logger.error(f"Error updating model configuration: {e}")

def main():
    """Download TheBloke's Mistral 7B GGUF model."""
    parser = argparse.ArgumentParser(description="Download TheBloke's Mistral 7B GGUF model")
    parser.add_argument("--model_id", type=str, default="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
                        help="Model ID on Hugging Face")
    parser.add_argument("--output_dir", type=str, default="models/mistral-7b-instruct", 
                        help="Output directory for the model")
    parser.add_argument("--model_file", type=str, default="mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
                       help="Specific model file to download")
    args = parser.parse_args()
    
    try:
        # Download the model
        file_path = download_gguf_model(args.model_id, args.output_dir, args.model_file)
        
        # Verify the model file
        if verify_model_files(args.output_dir, args.model_file):
            logger.info("Model file verification passed")
            
            # Update the configuration file
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                update_config_file(args.model_id, args.output_dir, args.model_file)
            except Exception as e:
                logger.warning(f"Could not update configuration: {e}")
                
            logger.info(f"Model download complete: {file_path}")
        else:
            logger.error("Model file verification failed")
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
"""
Mistral 7B model downloader.

This module handles downloading and verifying the Mistral 7B model from Hugging Face.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Handles downloading and verifying model files from Hugging Face."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model downloader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "math_llm_system", "models")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def download_mistral_7b(self, revision: str = "main") -> str:
        """
        Download the Mistral 7B model.
        
        Args:
            revision: Model revision to download (branch, tag, or commit hash)
            
        Returns:
            Path to the downloaded model
            
        Raises:
            ValueError: If download fails
        """
        model_id = "mistralai/Mistral-7B-v0.1"
        logger.info(f"Downloading Mistral 7B model from {model_id}...")
        
        try:
            # Download the model
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=False,
                token=os.environ.get("HF_TOKEN")  # In case the model requires authentication
            )
            
            logger.info(f"Model downloaded successfully to {model_path}")
            return model_path
            
        except (HFValidationError, RepositoryNotFoundError) as e:
            logger.error(f"Failed to download model: {e}")
            raise ValueError(f"Failed to download Mistral 7B model: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading model: {e}")
            raise ValueError(f"Unexpected error downloading Mistral 7B model: {e}")
    
    def download_tokenizer(self, model_id: str = "mistralai/Mistral-7B-v0.1") -> str:
        """
        Download the tokenizer for the model.
        
        Args:
            model_id: Model ID on Hugging Face
            
        Returns:
            Path to the downloaded tokenizer
        """
        logger.info(f"Downloading tokenizer from {model_id}...")
        
        try:
            # Download tokenizer files
            tokenizer_path = Path(self.cache_dir) / model_id
            os.makedirs(tokenizer_path, exist_ok=True)
            
            for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                hf_hub_download(
                    repo_id=model_id,
                    filename=file,
                    cache_dir=self.cache_dir,
                    token=os.environ.get("HF_TOKEN")
                )
            
            logger.info(f"Tokenizer downloaded successfully to {tokenizer_path}")
            return str(tokenizer_path)
            
        except Exception as e:
            logger.error(f"Failed to download tokenizer: {e}")
            raise ValueError(f"Failed to download tokenizer: {e}")
    
    def verify_model_files(self, model_path: str) -> bool:
        """
        Verify that all necessary model files are present.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if all files are present, False otherwise
        """
        required_files = [
            "config.json",
            "pytorch_model.bin" if not os.path.exists(os.path.join(model_path, "pytorch_model-00001-of-00003.bin")) else None,
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        # Remove None values (for sharded models)
        required_files = [f for f in required_files if f is not None]
        
        # Check for sharded model files
        if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            # Check if this is a sharded model
            import glob
            shards = glob.glob(os.path.join(model_path, "pytorch_model-*.bin"))
            if not shards:
                logger.error("No model weights found (neither single file nor shards)")
                return False
        
        # Check each required file
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file}")
                return False
        
        logger.info("All required model files are present")
        return True

"""
Inference module for Mistral 7B.

This module provides optimized inference for the Mistral 7B model using an external LMStudio server.
"""
import os
import logging
import requests
import json
from typing import List, Optional

logger = logging.getLogger(__name__)

class MistralInference:
    """Inference engine for Mistral 7B model using external LMStudio server."""
    
    def __init__(
        self,
        model_path: str = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        api_url: str = "http://127.0.0.1:1234"
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model file (not used with external server)
            n_ctx: Context window size
            n_threads: Number of threads to use (not used with external server)
            n_gpu_layers: Number of layers to offload to GPU (not used with external server)
            api_url: URL of the LMStudio API server
        """
        self.api_url = api_url
        logger.info(f"Using external LLM server at {api_url}")
        
        # Check if the server is available
        try:
            response = requests.get(f"{api_url}/v1/models")
            if response.status_code == 200:
                logger.info("Connected to LMStudio server successfully")
                models = response.json()
                logger.info(f"Available models: {models}")
            else:
                logger.warning(f"Server returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to LMStudio server: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from a prompt using the LMStudio API.
        
        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences to stop generation at
            
        Returns:
            Generated text
        """
        # Set up generation parameters
        payload = {
            "model": "mistral-7b-instruct-v0.3",
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens is not None else 2048,
            "temperature": temperature,
            "stop": stop_sequences if stop_sequences else None
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                output = response.json()
                generated_text = output["choices"][0]["text"].strip()
                return generated_text
            else:
                logger.error(f"Error generating response: HTTP {response.status_code}")
                logger.error(response.text)
                return f"Error: Server returned status code {response.status_code}"
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

"""
LMStudio server connection validator.

This module checks connectivity with the LMStudio server running the Mistral model.
"""
import os
import logging
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class LMStudioValidator:
    """Handles checking connectivity with the LMStudio server."""
    
    def __init__(self, api_url: str = "http://127.0.0.1:1234"):
        """
        Initialize the LMStudio validator.
        
        Args:
            api_url: URL of the LMStudio API server
        """
        self.api_url = api_url
        logger.info(f"LMStudio server URL: {api_url}")
    
    def check_connectivity(self) -> bool:
        """
        Check connectivity with the LMStudio server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        logger.info(f"Checking connectivity with LMStudio server at {self.api_url}...")
        
        try:
            response = requests.get(f"{self.api_url}/v1/models")
            
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Connected to LMStudio server successfully. Available models: {models}")
                return True
            else:
                logger.error(f"Failed to connect to LMStudio server: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to LMStudio server: {str(e)}")
            return False
    
    def verify_mistral_model_availability(self) -> bool:
        """
        Verify that the Mistral 7B model is available on the LMStudio server.
        
        Returns:
            True if the model is available, False otherwise
        """
        logger.info("Checking Mistral model availability...")
        
        try:
            response = requests.get(f"{self.api_url}/v1/models")
            
            if response.status_code == 200:
                models = response.json()
                
                # Check if Mistral model is in the list
                for model in models.get("data", []):
                    model_id = model.get("id", "")
                    if "mistral" in model_id.lower():
                        logger.info(f"Found Mistral model: {model_id}")
                        return True
                
                logger.warning("No Mistral model found in available models")
                return False
            else:
                logger.error(f"Failed to get models list: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
    
    def test_generation(self) -> bool:
        """
        Test generation with the Mistral model.
        
        Returns:
            True if generation is successful, False otherwise
        """
        logger.info("Testing generation with Mistral model...")
        
        try:
            payload = {
                "model": "mistral-7b-instruct-v0.3",
                "prompt": "<s>[INST] Hello, how are you? [/INST]",
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.api_url}/v1/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10
            )
            
            if response.status_code == 200:
                output = response.json()
                generated_text = output["choices"][0]["text"].strip()
                logger.info(f"Generation test successful. Response: {generated_text[:50]}...")
                return True
            else:
                logger.error(f"Generation test failed: HTTP {response.status_code}")
                logger.error(response.text)
                return False
                
        except Exception as e:
            logger.error(f"Error testing generation: {str(e)}")
            return False

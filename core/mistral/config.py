"""
Configuration settings for the Mistral 7B model via LMStudio.
"""

import os
from typing import Dict, Any

# LMStudio server settings
LMSTUDIO_URL = "http://127.0.0.1:1234"
MODEL_NAME = "mistral-7b-instruct-v0.3"  # Model name in LMStudio

# Inference settings
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1

# System prompt template for mathematical reasoning
MATH_SYSTEM_PROMPT = """
You are a mathematical reasoning assistant with expertise in algebra, calculus, 
geometry, statistics, and other mathematical fields. Your task is to solve 
mathematical problems step-by-step, explaining your reasoning clearly.

When processing a problem:
1. Identify the mathematical domain and relevant concepts
2. Formulate a solution strategy
3. Execute the solution step-by-step
4. Verify your answer and provide intuitive explanations
"""

def get_model_config() -> Dict[str, Any]:
    """
    Get the model configuration settings.
    
    Returns:
        Dictionary containing model configuration settings
    """
    return {
        "lmstudio_url": os.environ.get("LMSTUDIO_URL", LMSTUDIO_URL),
        "model_name": os.environ.get("MODEL_NAME", MODEL_NAME),
        "max_length": int(os.environ.get("MAX_LENGTH", DEFAULT_MAX_LENGTH)),
        "temperature": float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE)),
        "top_p": float(os.environ.get("TOP_P", DEFAULT_TOP_P)),
        "top_k": int(os.environ.get("TOP_K", DEFAULT_TOP_K)),
        "repetition_penalty": float(os.environ.get("REPETITION_PENALTY", DEFAULT_REPETITION_PENALTY)),
        "system_prompt": os.environ.get("MATH_SYSTEM_PROMPT", MATH_SYSTEM_PROMPT),
    }

def update_lmstudio_url(url: str) -> None:
    """
    Update the LMStudio server URL in the configuration file.
    
    Args:
        url: New LMStudio server URL to use
    """
    # Open the configuration file
    file_path = os.path.abspath(__file__)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Update the LMStudio URL
    for i, line in enumerate(lines):
        if line.startswith('LMSTUDIO_URL = '):
            lines[i] = f'LMSTUDIO_URL = "{url}"  # Updated LMStudio URL\n'
    
    # Write the updated configuration back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
        
    print(f"Updated LMStudio URL to {url} in config file")

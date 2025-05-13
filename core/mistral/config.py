"""
Configuration settings for the Mistral 7B model.
"""

import os
from typing import Dict, Any

# Model identification
# Using a small causal language model for development
MODEL_ID = "distilgpt2"  # Small, fast causal LM for development 
# MODEL_ID = "mistralai/Mistral-7B-v0.1"  # Original model for production
MODEL_REVISION = "main"

# File paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_MODEL_DIR, "cache")

# Inference settings
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1

# Quantization settings
QUANTIZATION_ENABLED = True
QUANTIZATION_BITS = 4  # 4-bit quantization

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
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dir": os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        "cache_dir": os.environ.get("CACHE_DIR", DEFAULT_CACHE_DIR),
        "max_length": int(os.environ.get("MAX_LENGTH", DEFAULT_MAX_LENGTH)),
        "temperature": float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE)),
        "top_p": float(os.environ.get("TOP_P", DEFAULT_TOP_P)),
        "top_k": int(os.environ.get("TOP_K", DEFAULT_TOP_K)),
        "repetition_penalty": float(os.environ.get("REPETITION_PENALTY", DEFAULT_REPETITION_PENALTY)),
        "quantization": {
            "enabled": os.environ.get("QUANTIZATION_ENABLED", QUANTIZATION_ENABLED),
            "bits": int(os.environ.get("QUANTIZATION_BITS", QUANTIZATION_BITS)),
        },
        "system_prompt": os.environ.get("MATH_SYSTEM_PROMPT", MATH_SYSTEM_PROMPT),
    }

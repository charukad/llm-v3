#!/usr/bin/env python3
"""
Test script to verify MPS (Metal Performance Shaders) acceleration with 128 layers.
This script initializes the inference model with Apple Metal GPU acceleration 
and 128 layers, then runs a test query to verify functionality.
"""

import os
import time
import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%03d",
)

logger = logging.getLogger(__name__)

# Set environment variables for MPS acceleration
os.environ["USE_MPS"] = "1"
os.environ["MODEL_LAYERS"] = "128"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE_INFERENCE_FASTPATH"] = "1"

# Import after setting environment variables
from core.agent.llm_agent import CoreLLMAgent
from core.mistral.inference import MistralInference

def main():
    """Main test function."""
    # Check MPS availability
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    logger.info(f"MPS (Metal Performance Shaders) available: {mps_available}")
    
    if not mps_available:
        logger.error("MPS is not available on this device. Cannot test 128-layer acceleration.")
        return
    
    # Initialize the LLM agent
    logger.info("Initializing CoreLLMAgent with 128 layers and MPS acceleration...")
    agent = CoreLLMAgent()
    
    # Test differential equation example
    test_query = "Solve the differential equation dy/dx = y - x and plot the solution. Explain the behavior of the solution as x increases."
    
    logger.info(f"Running test query: '{test_query}'")
    
    # Measure response time
    start_time = time.time()
    response = agent.generate_response(test_query)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Response generated in {elapsed_time:.2f} seconds")
    logger.info(f"Response sample (first 300 chars): {response[:300]}...")
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main() 
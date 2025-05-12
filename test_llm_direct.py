#!/usr/bin/env python3
"""
Test the Core LLM Agent directly.
"""
import sys
import logging
import json
import time
import threading
import gc
from core.agent.llm_agent import CoreLLMAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_with_timeout(llm_agent, prompt, timeout_seconds=60):
    """
    Run the generation with a timeout
    """
    result = {"success": False, "error": "Unknown error"}
    
    def target():
        nonlocal result
        try:
            logger.info(f"Starting generation with timeout of {timeout_seconds} seconds")
            start_time = time.time()
            
            # Generate response with minimal tokens and temp for testing
            logger.info("Starting llm_agent.generate_response call")
            response = llm_agent.generate_response(
                prompt=prompt,
                temperature=0.1,
                max_tokens=20  # Reduced for testing
            )
            
            end_time = time.time()
            logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
            result = response
        except Exception as e:
            logger.error(f"Exception during generation: {str(e)}", exc_info=True)
            result = {"success": False, "error": str(e)}
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    
    logger.info("Starting generation thread")
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        logger.error(f"Generation timed out after {timeout_seconds} seconds")
        return {"success": False, "error": f"Timed out after {timeout_seconds} seconds"}
    
    logger.info("Generation thread completed")
    return result

def test_llm_direct(query, timeout_seconds=60):
    """
    Test the CoreLLMAgent directly without going through the API.
    """
    logger.info(f"Initializing CoreLLMAgent for query: {query}")
    
    try:
        # Force garbage collection before initializing
        gc.collect()
        
        # Use simpler query
        if query == "What is 2+235?":
            logger.info("Using super simple query for testing")
            query = "2+2"
        
        # Initialize the LLM agent with MPS acceleration
        start_time = time.time()
        llm_agent = CoreLLMAgent({
            "model_path": "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "use_mps": True,
            "model_layers": 32
        })
        end_time = time.time()
        
        logger.info(f"CoreLLMAgent initialized successfully in {end_time - start_time:.2f} seconds")
        
        # Format a minimal prompt for testing
        prompt = f"Answer this question briefly: {query}"
        
        # Generate response with timeout
        logger.info("Generating response with timeout...")
        response = generate_with_timeout(llm_agent, prompt, timeout_seconds)
        
        logger.info(f"Response complete: {json.dumps(response, indent=2)}")
        return response
        
    except Exception as e:
        logger.error(f"Error testing LLM: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        # Force cleanup
        gc.collect()

if __name__ == "__main__":
    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
    
    # Get timeout from command line or use default (30 seconds)
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    logger.info(f"Running test with query: '{query}' and timeout: {timeout} seconds")
    
    result = test_llm_direct(query, timeout)
    
    if result.get("success", False):
        print("\nSUCCESS: Got a valid response from CoreLLM")
        print(f"Response: {result.get('response')}")
    else:
        print("\nFAILURE: Could not get a valid response from CoreLLM")
        print(f"Error: {result.get('error', 'Unknown error')}") 
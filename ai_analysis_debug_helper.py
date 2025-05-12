#!/usr/bin/env python3
"""
Helper script for improving error handling and diagnostics in the AI analysis agent.
"""

import logging
import json
import time
import argparse
import traceback
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"ai_analysis_debug.log")
    ]
)

logger = logging.getLogger("ai_analysis_debug")

class LLMConnectionTester:
    """
    Test the connection between the AI analysis agent and the LLM.
    """
    
    def __init__(self):
        """Initialize the tester."""
        self.llm_agent = None
        
    def initialize(self):
        """Initialize the LLM agent."""
        try:
            # Try to import the CoreLLMAgent
            logger.info("Attempting to initialize CoreLLMAgent")
            
            # Import the module
            try:
                from core.agent.llm_agent import CoreLLMAgent
                logger.info("Successfully imported CoreLLMAgent module")
            except ImportError as e:
                logger.error(f"Failed to import CoreLLMAgent: {e}")
                return False
            
            # Try to get the agent from the server
            try:
                logger.info("Trying to get agent from server.py")
                from api.rest.server import get_core_llm_agent
                self.llm_agent = get_core_llm_agent()
                if self.llm_agent:
                    logger.info("Successfully retrieved LLM agent from server")
                    return True
                else:
                    logger.warning("get_core_llm_agent() returned None")
            except Exception as e:
                logger.error(f"Error getting agent from server: {e}")
                
            # Try to initialize a new agent directly
            try:
                logger.info("Trying to initialize a new CoreLLMAgent instance")
                
                # Get environment settings (similar to server.py)
                import os
                use_mps = os.environ.get("USE_MPS", "0") == "1"
                model_layers = int(os.environ.get("MODEL_LAYERS", "32"))
                model_path = os.environ.get("MODEL_PATH", "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
                
                # Create a new instance
                self.llm_agent = CoreLLMAgent({
                    "model_path": model_path,
                    "use_mps": use_mps,
                    "model_layers": model_layers
                })
                
                logger.info(f"Successfully created new LLM agent with model: {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error initializing new CoreLLMAgent: {e}")
                traceback.print_exc()
                
            return False
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            traceback.print_exc()
            return False
    
    def test_generate_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Test the generate_response method of the LLM agent.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The response from the LLM, or None if there was an error
        """
        if self.llm_agent is None:
            logger.error("LLM agent is not initialized")
            return None
            
        try:
            logger.info(f"Testing generate_response with prompt: {prompt[:50]}...")
            
            # Test if the method exists
            if not hasattr(self.llm_agent, 'generate_response'):
                logger.error("LLM agent does not have generate_response method")
                logger.info(f"Available methods: {dir(self.llm_agent)}")
                return None
                
            # Call the method with timing
            start_time = time.time()
            logger.info("Calling generate_response")
            
            response = self.llm_agent.generate_response(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1024
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Response received in {duration:.2f} seconds")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response: {json.dumps(response, default=str)[:200]}...")
            
            return response
        except Exception as e:
            logger.error(f"Error calling generate_response: {e}")
            traceback.print_exc()
            return None
            
    def test_all_methods(self, prompt: str):
        """
        Test all available methods for the LLM agent.
        
        Args:
            prompt: The prompt to test with
        """
        if self.llm_agent is None:
            logger.error("LLM agent is not initialized")
            return
            
        # List of methods to try
        methods = [
            'generate_response',
            'generate',
            'generate_text',
            'process_prompt',
            '__call__'
        ]
        
        for method_name in methods:
            if hasattr(self.llm_agent, method_name):
                logger.info(f"Testing method: {method_name}")
                
                try:
                    start_time = time.time()
                    
                    if method_name == 'generate_response':
                        response = self.llm_agent.generate_response(
                            prompt=prompt,
                            temperature=0.1,
                            max_tokens=1024
                        )
                    elif method_name == 'generate':
                        response = self.llm_agent.generate(prompt)
                    elif method_name == 'generate_text':
                        response = self.llm_agent.generate_text(prompt, temperature=0.1, max_tokens=1024)
                    elif method_name == 'process_prompt':
                        response = self.llm_agent.process_prompt({"prompt": prompt})
                    elif method_name == '__call__':
                        response = self.llm_agent(prompt)
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    logger.info(f"Response from {method_name} received in {duration:.2f} seconds")
                    logger.info(f"Response type: {type(response)}")
                    logger.info(f"Response: {json.dumps(response, default=str)[:200]}...")
                except Exception as e:
                    logger.error(f"Error calling {method_name}: {e}")
                    traceback.print_exc()
            else:
                logger.info(f"Method not available: {method_name}")

def patch_ai_analysis_agent():
    """
    Generate code to patch the AI analysis agent with improved error handling.
    """
    improved_code = '''
# Replace the _run_with_timeout method with this improved version
async def _run_with_timeout(self, prompt: str) -> Dict[str, Any]:
    """
    Run the LLM with a timeout to prevent hanging.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        The LLM response
    """
    loop = asyncio.get_event_loop()
    try:
        logger.info("Starting LLM generation with timeout")
        # Record message size to help with diagnostics
        token_estimate = len(prompt.split()) # Rough estimate
        logger.info(f"Prompt length estimate: {token_estimate} tokens")
        
        # Try to use a more detailed error handler
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: self.llm_agent.generate_response(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1024
                )
            ),
            timeout=25.0  # 25 second timeout
        )
        
        logger.info("LLM generation completed successfully")
        return response
    except asyncio.TimeoutError:
        logger.error("LLM generation timed out")
        raise
    except Exception as e:
        logger.error(f"Error during LLM generation: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        # Include stack trace for debugging
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
'''
    
    logger.info("Generated improved error handling code for _run_with_timeout")
    logger.info(improved_code)
    
    return improved_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Analysis Debug Helper')
    parser.add_argument('--test', action='store_true', help='Test LLM connection')
    parser.add_argument('--patch', action='store_true', help='Generate improved error handling code')
    parser.add_argument('--prompt', type=str, default="What is 2+2?", help='Test prompt')
    
    args = parser.parse_args()
    
    if args.test:
        tester = LLMConnectionTester()
        logger.info("Initializing LLM connection tester")
        
        if tester.initialize():
            logger.info("Testing LLM connection with a simple prompt")
            response = tester.test_generate_response(args.prompt)
            
            if response:
                logger.info("LLM connection test successful")
                logger.info("Testing all available methods")
                tester.test_all_methods(args.prompt)
            else:
                logger.error("LLM connection test failed")
        else:
            logger.error("Failed to initialize LLM connection tester")
    
    if args.patch:
        logger.info("Generating improved error handling code")
        patch_ai_analysis_agent() 
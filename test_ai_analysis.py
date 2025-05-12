#!/usr/bin/env python3
"""
Test script to send a prompt to the AI analysis endpoint and process the results.
"""
import requests
import json
import sys
import logging
import time
import traceback
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API endpoints
API_BASE_URL = "http://localhost:8000"
ANALYSIS_ENDPOINT = f"{API_BASE_URL}/ai-analysis/analyze"
MATH_ENDPOINT = f"{API_BASE_URL}/math/query"
VISUALIZATION_ENDPOINT = f"{API_BASE_URL}/visualization/generate"
CORE_LLM_ENDPOINT = f"{API_BASE_URL}/workflow/process/text"

def retry_request(func, max_retries=3, retry_delay=2):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Failed after {max_retries} retries: {str(e)}")
                    raise
                wait_time = retry_delay * (2 ** (retries - 1))
                logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
    return wrapper

@retry_request
def analyze_prompt(prompt):
    """Send prompt to analysis endpoint to determine which agent to use."""
    logger.info(f"Sending prompt to analysis endpoint: {prompt[:50]}...")
    
    try:
        response = requests.post(
            ANALYSIS_ENDPOINT,
            json={"query": prompt},
            timeout=400  # 30 second timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Error from analysis endpoint: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {
                "success": False,
                "error": f"Analysis endpoint returned status code: {response.status_code}"
            }
        
        analysis_result = response.json()
        logger.info(f"Analysis complete. Primary agent: {analysis_result.get('analysis', {}).get('routing', {}).get('primary_agent', 'unknown')}")
        
        return analysis_result
    
    except RequestException as e:
        logger.error(f"Request error sending prompt to analysis endpoint: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error sending prompt to analysis endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@retry_request
def route_to_agent(prompt, analysis):
    """Route the prompt to the appropriate agent based on the analysis."""
    if not analysis.get("success", False):
        logger.error("Analysis was unsuccessful, cannot route to agent")
        return {
            "success": False,
            "error": "Analysis was unsuccessful"
        }
    
    try:
        # Get routing information
        routing = analysis.get("analysis", {}).get("routing", {})
        primary_agent = routing.get("primary_agent", "core_llm_agent")
        
        logger.info(f"Routing prompt to agent: {primary_agent}")
        
        # Route to the appropriate agent
        if primary_agent == "math_computation_agent":
            return send_to_math_agent(prompt)
        elif primary_agent == "visualization_agent":
            return send_to_visualization_agent(prompt)
        else:  # Default to core_llm_agent
            return send_to_core_llm_agent(prompt)
    
    except Exception as e:
        logger.error(f"Error routing prompt to agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@retry_request
def send_to_math_agent(prompt):
    """Send prompt to math computation agent."""
    logger.info(f"Sending prompt to math computation agent: {prompt[:50]}...")
    
    try:
        response = requests.post(
            MATH_ENDPOINT,
            json={"query": prompt},
            timeout=30  # 30 second timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Error from math agent: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {
                "success": False,
                "error": f"Math agent returned status code: {response.status_code}"
            }
        
        return response.json()
    
    except RequestException as e:
        logger.error(f"Request error sending prompt to math agent: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error sending prompt to math agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@retry_request
def send_to_visualization_agent(prompt):
    """Send prompt to visualization agent."""
    logger.info(f"Sending prompt to visualization agent: {prompt[:50]}...")
    
    try:
        response = requests.post(
            VISUALIZATION_ENDPOINT,
            json={"query": prompt},
            timeout=30  # 30 second timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Error from visualization agent: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {
                "success": False,
                "error": f"Visualization agent returned status code: {response.status_code}"
            }
        
        return response.json()
    
    except RequestException as e:
        logger.error(f"Request error sending prompt to visualization agent: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error sending prompt to visualization agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@retry_request
def send_to_core_llm_agent(prompt):
    """Send prompt to core LLM agent."""
    logger.info(f"Sending prompt to core LLM agent: {prompt[:50]}...")
    
    try:
        response = requests.post(
            CORE_LLM_ENDPOINT,
            json={
                "text": prompt,
                "context_id": None,
                "conversation_id": None,
                "workflow_options": {"generate_visualization": False}
            },
            timeout=30  # 30 second timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Error from core LLM agent: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {
                "success": False,
                "error": f"Core LLM agent returned status code: {response.status_code}"
            }
        
        return response.json()
    
    except RequestException as e:
        logger.error(f"Request error sending prompt to core LLM agent: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error sending prompt to core LLM agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def process_prompt(prompt):
    """
    Process a prompt by:
    1. Sending it to the analysis endpoint to determine the appropriate agent
    2. Routing the prompt to the selected agent
    3. Returning the results
    """
    try:
        # Step 1: Try to get health status first
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            logger.info(f"Server health check: {health_response.status_code}")
            if health_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Server health check failed: {health_response.status_code}"
                }
        except Exception as e:
            logger.error(f"Server health check failed: {str(e)}")
            return {
                "success": False,
                "error": f"Server health check failed: {str(e)}"
            }
            
        # Step 2: Analyze the prompt
        analysis_result = analyze_prompt(prompt)
        
        if not analysis_result.get("success", False):
            return {
                "success": False,
                "error": analysis_result.get("error", "Unknown error during analysis")
            }
        
        # Step 3: Route to the appropriate agent
        agent_result = route_to_agent(prompt, analysis_result)
        
        # Step 4: Return combined results
        return {
            "success": agent_result.get("success", False),
            "analysis": analysis_result.get("analysis", {}),
            "response": agent_result,
            "prompt": prompt
        }
    except Exception as e:
        logger.error(f"Error in process_prompt: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": f"Failed to process prompt: {str(e)}",
            "prompt": prompt
        }

if __name__ == "__main__":
    # Get prompt from command line or use default
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2+2?"
    
    logger.info(f"Processing prompt: {prompt}")
    
    start_time = time.time()
    result = process_prompt(prompt)
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print("PROMPT PROCESSING RESULTS")
    print("="*50)
    
    print(f"\nPrompt: {prompt}")
    
    if result.get("success", False):
        print("\nAnalysis:")
        analysis = result.get("analysis", {})
        print(f"- Primary Agent: {analysis.get('routing', {}).get('primary_agent', 'unknown')}")
        print(f"- Operations: {', '.join(analysis.get('operations', []))}")
        print(f"- Concepts: {', '.join(analysis.get('concepts', []))}")
        print(f"- Complexity: {analysis.get('complexity', 'unknown')}")
        
        print("\nResponse:")
        response = result.get("response", {})
        
        # Handle different response formats from different agents
        if "answer" in response:
            print(f"Answer: {response.get('answer')}")
        elif "response" in response:
            print(f"Response: {response.get('response')}")
        elif "result" in response:
            print(f"Result: {response.get('result')}")
        elif "output" in response:  # Added for workflow endpoint format
            print(f"Output: {response.get('output')}")
        else:
            print(json.dumps(response, indent=2))
    else:
        print(f"\nERROR: {result.get('error', 'Unknown error')}")
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("="*50) 
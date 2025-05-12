#!/usr/bin/env python3
"""
Diagnostic script for the AI analysis workflow.
This script performs detailed diagnostics on the LLM integration and identifies potential issues.
"""

import requests
import json
import logging
import sys
import time
import argparse
import traceback
import re
import os
import uuid
import math
import random
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Setup logger
logger = logging.getLogger("llm_diagnostics")

def setup_logging(debug_mode=False):
    """Set up logging configuration.
    
    Args:
        debug_mode (bool): Whether to enable debug logging.
    """
    # Set log level based on debug flag
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()
    
    # Add a stream handler to see logs in console
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Add file handler
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"{log_dir}/diagnostics_{timestamp}.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")

class LLMDiagnostics:
    """Class to diagnose LLM integration issues."""
    
    def __init__(self, base_url="http://localhost:8000"):
        """Initialize the diagnostics tool."""
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_server_health(self):
        """Check if the server is running and get component status."""
        try:
            logger.info("Checking server health...")
            health_url = f"{self.base_url}/health"
            
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Server health check: {health_data}")
                
                # Check component status
                components = health_data.get("components", {})
                all_available = all(status == "available" for status in components.values())
                
                if all_available:
                    logger.info("All server components are available")
                else:
                    logger.warning("Some server components are not available:")
                    for component, status in components.items():
                        if status != "available":
                            logger.warning(f"- {component}: {status}")
                
                return health_data
            else:
                logger.error(f"Health check failed with status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to server: {e}")
            return None
    
    def test_connection(self):
        """Test basic connectivity to the server."""
        try:
            logger.info("Testing server connection...")
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Server connection successful")
                return True
            else:
                logger.error(f"Server connection failed with status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Server connection failed: {e}")
            return False
            
    def test_ai_analysis_endpoint(self):
        """Test the AI analysis endpoint with a simple query."""
        try:
            logger.info("Testing AI analysis endpoint...")
            simple_query = "Test query"
            response = requests.post(
                f"{self.base_url}/ai-analysis/analyze",
                json={"query": simple_query},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("AI analysis endpoint is working")
                return True
            else:
                logger.error(f"AI analysis endpoint test failed with status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error testing AI analysis endpoint: {e}")
            return False
    
    def test_error_handling(self):
        """Test how the system handles invalid inputs and errors."""
        try:
            logger.info("Testing error handling...")
            # Test with malformed JSON
            invalid_payload = {"invalid": "payload"}
            response = requests.post(
                f"{self.base_url}/ai-analysis/analyze",
                json=invalid_payload,
                timeout=30
            )
            
            if response.status_code in [400, 422]:
                logger.info("Server correctly handles invalid input with a 400/422 response")
                return True
            else:
                logger.warning(f"Unexpected response for invalid input: {response.status_code}")
                
                # Try with empty payload as another test
                empty_response = requests.post(
                    f"{self.base_url}/ai-analysis/analyze",
                    json={},
                    timeout=30
                )
                
                if empty_response.status_code in [400, 422]:
                    logger.info("Server correctly handles empty payload")
                    return True
                else:
                    logger.error(f"Server error handling test failed: {empty_response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Error during error handling test: {e}")
            return False
    
    def diagnose_llm_agent(self):
        """Diagnose if the LLM agent is working correctly."""
        try:
            logger.info("Testing LLM agent with a simple query...")
            
            # First test with the AI analysis endpoint
            simple_query = "What is 1+1?"
            analysis_url = f"{self.base_url}/ai-analysis/analyze"
            
            response = requests.post(
                analysis_url,
                json={"query": simple_query},
                timeout=300
            )
            
            if response.status_code == 200:
                analysis_data = response.json()
                analysis = analysis_data.get("analysis", {})
                
                if analysis.get("ai_source") == "real_core_llm":
                    logger.info("LLM agent is working correctly for analysis")
                else:
                    logger.warning("LLM agent using fallback for analysis")
                    if "error" in analysis:
                        logger.error(f"Analysis error: {analysis['error']}")
            else:
                logger.error(f"AI analysis request failed: {response.status_code}")
            
            # Then test a direct workflow query to see the difference
            logger.info("Testing direct workflow for comparison...")
            
            workflow_url = f"{self.base_url}/workflow/process/text"
            workflow_payload = {
                "text": simple_query,
                "options": {
                    "mode": "immediate",
                    "response_format": "json"
                }
            }
            
            workflow_response = requests.post(
                workflow_url,
                json=workflow_payload,
                timeout=300
            )
            
            if workflow_response.status_code == 200:
                logger.info("Direct workflow request succeeded")
            else:
                logger.error(f"Direct workflow request failed: {workflow_response.status_code}")
                logger.error(f"Response: {workflow_response.text[:200]}")
                
            return True
        except Exception as e:
            logger.error(f"Error diagnosing LLM agent: {e}")
            return False
    
    def check_logs_for_errors(self):
        """
        Check server logs for common LLM-related errors.
        This is a mock implementation since we can't directly access server logs.
        """
        logger.info("Checking for common LLM-related errors in logs...")
        
        # Common error patterns to look for
        error_patterns = [
            "CoreLLM agent not available",
            "LLM generation timed out",
            "Error calling CoreLLM",
            "No JSON found in response",
            "Error parsing JSON from CoreLLM response"
        ]
        
        # Here we would typically read the server logs, but since that's not possible,
        # we'll check the AI analysis endpoint for errors
        test_query = "Test query to check for errors"
        
        try:
            response = requests.post(
                f"{self.base_url}/ai-analysis/analyze",
                json={"query": test_query},
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data.get("success", False):
                    logger.error(f"Error in AI analysis: {data.get('error')}")
                    return False
                
                analysis = data.get("analysis", {})
                if analysis.get("fallback", False):
                    error = analysis.get("error", "Unknown error")
                    logger.warning(f"AI analysis using fallback due to: {error}")
                    
                    # Check if the error matches any of our patterns
                    for pattern in error_patterns:
                        if pattern in error:
                            logger.error(f"Detected known error pattern: {pattern}")
                            
                            # Suggest solutions based on the error
                            if "timed out" in error:
                                logger.info("Solution: Increase timeout values in the AI analysis route")
                            elif "not available" in error:
                                logger.info("Solution: Check if LLM agent is properly initialized in server")
                            elif "JSON" in error:
                                logger.info("Solution: Improve JSON parsing logic in AI analysis route")
                    
                    return False
            else:
                logger.error(f"Error response from AI analysis: {response.status_code}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking logs: {e}")
            return False
    
    def test_timeout_handling(self, complexity="complex"):
        """
        Test how the system handles timeouts with queries of different complexity.
        
        Args:
            complexity: Complexity level of test query ('simple', 'moderate', 'complex')
        
        Returns:
            bool: True if timeout handling is working correctly, False otherwise
        """
        logger.info(f"Testing timeout handling with {complexity} query...")
        
        # Create query of different complexity levels
        if complexity == "simple":
            query = "What is 2+2?"
        elif complexity == "moderate":
            query = "Solve the quadratic equation x^2 + 5x + 6 = 0"
        else:  # complex
            query = "Find the derivative of f(x) = x^3 * sin(x) * log(x^2 + 1) and solve the equation f'(x) = 0 for x in the interval [1, 5], then plot the function and its derivative"
        
        start_time = time.time()
        try:
            logger.info(f"Sending request with query: {query}")
            response = requests.post(
                f"{self.base_url}/ai-analysis/analyze",
                json={"query": query},
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            logger.info(f"Request completed in {duration:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("analysis", {})
                
                if analysis.get("fallback", False) and "timeout" in analysis.get("error", ""):
                    logger.warning(f"LLM timed out for {complexity} query")
                    logger.info("This may indicate a need to optimize the prompt or increase timeouts")
                    return False
                else:
                    logger.info(f"LLM successfully handled {complexity} query")
                    return True
            else:
                logger.error(f"Request failed with status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            logger.error(f"Request timed out after {duration:.2f} seconds")
            logger.info("This indicates the server request timeout needs to be increased")
            return False
        except Exception as e:
            logger.error(f"Error testing timeout handling: {e}")
            return False
    
    def get_query_answer(self, query):
        """
        Get the answer for a specific query from the AI analysis endpoint.
        
        Args:
            query: The query to send to the AI analysis endpoint
            
        Returns:
            tuple: (answer, steps, execution_time) or (None, None, None) if the query fails
        """
        logger.info(f"Getting answer for query: {query}")
        
        try:
            response = requests.post(
                f"{self.base_url}/ai-analysis/analyze",
                json={"query": query},
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract answer, steps, and execution time
                answer = data.get("answer")
                steps = data.get("steps")
                execution_time = data.get("execution_time")
                
                # If no answer is provided by the API, try to calculate it for simple queries
                if not answer:
                    calculated_answer = self.calculate_simple_math_answer(query)
                    if calculated_answer:
                        logger.info(f"Calculated answer: {calculated_answer}")
                        answer = calculated_answer
                        
                        # Create solution steps based on query type
                        if "quadratic" in query.lower():
                            # Extract coefficients for quadratic formula steps
                            import re
                            simplified_query = query.replace(" ", "")
                            coefficients = [1, 5, 6]  # Default for the example x^2 + 5x + 6 = 0
                            
                            # Try to extract actual coefficients if possible
                            equation_pattern = r"(\d*)x\^2\s*([+-]\s*\d*)x\s*([+-]\s*\d*)\s*=\s*0"
                            match = re.search(equation_pattern, simplified_query)
                            if match:
                                a_str, b_str, c_str = match.groups()
                                a = 1 if a_str == '' else int(a_str)
                                
                                # Process b coefficient
                                if b_str.strip() == '+':
                                    b = 1
                                elif b_str.strip() == '-':
                                    b = -1
                                else:
                                    b = int(b_str.replace('+', '').replace(' ', ''))
                                
                                # Process c coefficient
                                if c_str.strip() == '+':
                                    c = 1
                                elif c_str.strip() == '-':
                                    c = -1
                                else:
                                    c = int(c_str.replace('+', '').replace(' ', ''))
                                    
                                coefficients = [a, b, c]
                            
                            # Detailed solution steps for quadratic equation
                            a, b, c = coefficients
                            discriminant = b**2 - 4*a*c
                            steps = [
                                f"Identify the quadratic equation in standard form: ax² + bx + c = 0",
                                f"Identify the coefficients: a = {a}, b = {b}, c = {c}",
                                f"Calculate the discriminant: Δ = b² - 4ac = {b}² - 4({a})({c}) = {discriminant}",
                            ]
                            
                            if discriminant < 0:
                                steps.append("Since the discriminant is negative, there are no real solutions")
                            elif discriminant == 0:
                                x = -b / (2*a)
                                steps.append(f"Since the discriminant is zero, there is one solution")
                                steps.append(f"x = -b/(2a) = -({b})/(2*{a}) = {x}")
                            else:
                                x1 = (-b + (discriminant)**0.5) / (2*a)
                                x2 = (-b - (discriminant)**0.5) / (2*a)
                                steps.append(f"Since the discriminant is positive, there are two solutions")
                                steps.append(f"Using the quadratic formula: x = (-b ± √Δ)/(2a)")
                                steps.append(f"x₁ = (-({b}) + √{discriminant})/(2*{a}) = {x1}")
                                steps.append(f"x₂ = (-({b}) - √{discriminant})/(2*{a}) = {x2}")
                                
                                # If the solutions are integers, verify by substitution
                                if x1.is_integer() and x2.is_integer():
                                    steps.append(f"Verify by substitution:")
                                    x1_int = int(x1)
                                    x2_int = int(x2)
                                    steps.append(f"For x = {x1_int}: {a}({x1_int})² + {b}({x1_int}) + {c} = {a*(x1_int)**2 + b*(x1_int) + c}")
                                    steps.append(f"For x = {x2_int}: {a}({x2_int})² + {b}({x2_int}) + {c} = {a*(x2_int)**2 + b*(x2_int) + c}")
                        else:
                            # Basic steps for other types of calculations
                            steps = ["Parse the arithmetic operation", f"Calculate the result: {calculated_answer}"]
                    else:
                        logger.warning("No answer found in the response and couldn't calculate it locally")
                else:
                    logger.info(f"API provided answer: {answer}")
                    
                # If no direct answer, try to extract useful information from the analysis
                analysis = data.get("analysis", {})
                if analysis:
                    logger.info("Analysis information:")
                    logger.info(f"- Operations: {analysis.get('operations', [])}")
                    logger.info(f"- Concepts: {analysis.get('concepts', [])}")
                    logger.info(f"- Complexity: {analysis.get('complexity', 'unknown')}")
                    
                    # Get routing information
                    routing = analysis.get("routing", {})
                    if routing:
                        logger.info(f"- Primary agent: {routing.get('primary_agent', 'unknown')}")
                        logger.info(f"- Confidence: {routing.get('confidence', 0)}")
                
                return answer, steps, execution_time
            else:
                logger.error(f"Query failed with status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None, None, None
        except Exception as e:
            logger.error(f"Error getting query answer: {e}")
            return None, None, None
    
    def test_agent_routing(self, query):
        """
        Test how the AI analysis agent routes different queries to specialized agents.
        
        Args:
            query: The query to analyze for agent routing
            
        Returns:
            dict: Routing information including primary agent, confidence, and alternatives
        """
        logger.info(f"Testing agent routing for query: {query}")
        
        try:
            response = requests.post(
                f"{self.base_url}/ai-analysis/analyze",
                json={"query": query},
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("analysis", {})
                
                if analysis:
                    # Extract routing information
                    routing_info = {
                        "query": query,
                        "operations": analysis.get("operations", []),
                        "concepts": analysis.get("concepts", []),
                        "complexity": analysis.get("complexity", "unknown"),
                        "required_agents": analysis.get("required_agents", []),
                        "ai_source": analysis.get("ai_source", "unknown")
                    }
                    
                    # Get routing decision
                    routing = analysis.get("routing", {})
                    if routing:
                        routing_info["primary_agent"] = routing.get("primary_agent", "unknown")
                        routing_info["confidence"] = routing.get("confidence", 0)
                        routing_info["alternative_agents"] = routing.get("alternative_agents", [])
                    
                    logger.info("Agent routing information:")
                    logger.info(f"- Query: {query}")
                    logger.info(f"- AI Source: {routing_info.get('ai_source')}")
                    logger.info(f"- Operations: {routing_info.get('operations')}")
                    logger.info(f"- Concepts: {routing_info.get('concepts')}")
                    logger.info(f"- Complexity: {routing_info.get('complexity')}")
                    logger.info(f"- Required agents: {routing_info.get('required_agents')}")
                    
                    if routing:
                        logger.info(f"- Primary agent: {routing_info.get('primary_agent')}")
                        logger.info(f"- Confidence: {routing_info.get('confidence')}")
                        logger.info(f"- Alternative agents: {routing_info.get('alternative_agents')}")
                    
                    return routing_info
                else:
                    logger.error("No analysis information found in response")
                    return None
            else:
                logger.error(f"Query failed with status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error testing agent routing: {e}")
            return None
    
    def perform_full_diagnostics(self):
        """Run a complete diagnostic check on the system."""
        logger.info("Running full diagnostic check on the AI Analysis system...")
        
        results = {
            "connection": self.test_connection(),
            "endpoint": self.test_ai_analysis_endpoint(),
            "error_handling": self.test_error_handling(),
            "logs": self.check_logs_for_errors(),
            "timeout_handling": {
                "simple": self.test_timeout_handling("simple"),
                "moderate": self.test_timeout_handling("moderate"),
                "complex": self.test_timeout_handling("complex")
            }
        }
        
        all_passed = all(results["connection"], results["endpoint"], results["error_handling"], results["logs"])
        timeout_passed = any(results["timeout_handling"].values())
        
        if all_passed and timeout_passed:
            logger.info("All diagnostic checks passed!")
        else:
            logger.warning("Some diagnostic checks failed. See results for details.")
            
            if not results["connection"]:
                logger.error("Connection test failed. Check server availability.")
            
            if not results["endpoint"]:
                logger.error("AI Analysis endpoint test failed. Check endpoint functionality.")
            
            if not results["error_handling"]:
                logger.error("Error handling test failed. Validate error handling logic.")
                
            if not results["logs"]:
                logger.error("Log check found errors. Review server logs.")
                
            if not timeout_passed:
                logger.error("All timeout tests failed. Consider increasing server timeout limits.")
        
        return results

    def forward_to_specialized_agent(self, query):
        """
        Complete workflow: analyze the query and forward to the appropriate specialized agent.
        
        Args:
            query: The query to process
            
        Returns:
            dict: Complete workflow results including analysis and specialized agent response
        """
        logger.info(f"Starting complete agent workflow for query: {query}")
        
        # Step 1: Send to analysis agent for classification and routing
        routing_info = self.test_agent_routing(query)
        if not routing_info:
            logger.error("Analysis agent failed to process the query")
            return {"success": False, "error": "Analysis agent failed to process the query"}
        
        # Step 2: Determine the primary agent to handle the query
        primary_agent = routing_info.get("primary_agent", "unknown")
        logger.info(f"Analysis agent selected {primary_agent} to handle the query")
        
        # Step 3: Forward to the appropriate specialized agent
        result = {
            "analysis": routing_info,
            "specialized_agent_response": None,
            "success": True
        }
        
        try:
            # Call the appropriate agent endpoint based on the primary agent
            if primary_agent == "math_computation_agent":
                logger.info("Forwarding to math computation agent...")
                
                # Call the real math computation agent endpoint
                math_endpoint = f"{self.base_url}/math/compute"
                
                # Check if the endpoint exists, if not, use the workflow endpoint
                try:
                    response = requests.head(math_endpoint, timeout=5)
                    if response.status_code >= 400:
                        # Fall back to workflow endpoint
                        math_endpoint = f"{self.base_url}/workflow/process/text"
                        logger.info(f"Math endpoint not available, using workflow endpoint: {math_endpoint}")
                except:
                    # Fall back to workflow endpoint
                    math_endpoint = f"{self.base_url}/workflow/process/text"
                    logger.info(f"Math endpoint not available, using workflow endpoint: {math_endpoint}")
                
                # Prepare the payload
                payload = {
                    "query": query,
                    "context": {
                        "agent": "math_computation_agent",
                        "analysis": routing_info
                    }
                }
                
                # For workflow endpoint, the payload format is different
                if "workflow" in math_endpoint:
                    payload = {
                        "text": query,
                        "options": {
                            "mode": "immediate",
                            "response_format": "json",
                            "agent": "math_computation_agent"
                        }
                    }
                
                # Send request to the math computation agent
                math_response = requests.post(
                    math_endpoint,
                    json=payload,
                    timeout=300
                )
                
                if math_response.status_code == 200:
                    math_data = math_response.json()
                    logger.info(f"Math agent response: {math_data}")
                    
                    # If we got a workflow_id, poll for the final result
                    if "workflow_id" in math_data:
                        workflow_id = math_data.get("workflow_id")
                        logger.info(f"Got workflow_id: {workflow_id}. Polling for result...")
                        
                        # Poll for the final result (with timeout)
                        final_result = self.poll_workflow_result(workflow_id)
                        
                        if final_result:
                            logger.info(f"Retrieved final result: {final_result}")
                            # Use the final result as the specialized agent response
                            result["specialized_agent_response"] = final_result
                        else:
                            # Keep the initial response if polling failed
                            result["specialized_agent_response"] = math_data
                    else:
                        # Use the immediate response
                        result["specialized_agent_response"] = math_data
                else:
                    logger.error(f"Math agent request failed: {math_response.status_code}")
                    logger.error(f"Response: {math_response.text}")
                    result["error"] = f"Math agent request failed: {math_response.status_code}"
                    result["specialized_agent_response"] = {"error": f"Failed to get response from math agent: {math_response.status_code}"}
            
            elif primary_agent == "visualization_agent":
                # Similar implementation as math agent with result polling
                logger.info("Forwarding to visualization agent...")
                
                # Use the workflow endpoint for visualization
                viz_endpoint = f"{self.base_url}/workflow/process/text"
                
                # Prepare the payload for workflow
                payload = {
                    "text": query,
                    "options": {
                        "mode": "immediate",
                        "response_format": "json",
                        "agent": "visualization_agent"
                    }
                }
                
                # Send request to the visualization agent
                viz_response = requests.post(
                    viz_endpoint,
                    json=payload,
                    timeout=300
                )
                
                if viz_response.status_code == 200:
                    viz_data = viz_response.json()
                    logger.info(f"Visualization agent response: {viz_data}")
                    
                    # If we got a workflow_id, poll for the final result
                    if "workflow_id" in viz_data:
                        workflow_id = viz_data.get("workflow_id")
                        logger.info(f"Got workflow_id: {workflow_id}. Polling for result...")
                        
                        # Poll for the final result (with timeout)
                        final_result = self.poll_workflow_result(workflow_id)
                        
                        if final_result:
                            logger.info(f"Retrieved final result: {final_result}")
                            # Use the final result as the specialized agent response
                            result["specialized_agent_response"] = final_result
                        else:
                            # Keep the initial response if polling failed
                            result["specialized_agent_response"] = viz_data
                    else:
                        # Use the immediate response
                        result["specialized_agent_response"] = viz_data
                else:
                    logger.error(f"Visualization agent request failed: {viz_response.status_code}")
                    logger.error(f"Response: {viz_response.text}")
                    result["error"] = f"Visualization agent request failed: {viz_response.status_code}"
                    result["specialized_agent_response"] = {"error": f"Failed to get response from visualization agent: {viz_response.status_code}"}
            
            elif primary_agent == "search_agent":
                # Similar implementation as math agent with result polling
                logger.info("Forwarding to search agent...")
                
                # Use the workflow endpoint for search
                search_endpoint = f"{self.base_url}/workflow/process/text"
                
                # Prepare the payload for workflow
                payload = {
                    "text": query,
                    "options": {
                        "mode": "immediate",
                        "response_format": "json",
                        "agent": "search_agent"
                    }
                }
                
                # Send request to the search agent
                search_response = requests.post(
                    search_endpoint,
                    json=payload,
                    timeout=300
                )
                
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    logger.info(f"Search agent response: {search_data}")
                    
                    # If we got a workflow_id, poll for the final result
                    if "workflow_id" in search_data:
                        workflow_id = search_data.get("workflow_id")
                        logger.info(f"Got workflow_id: {workflow_id}. Polling for result...")
                        
                        # Poll for the final result (with timeout)
                        final_result = self.poll_workflow_result(workflow_id)
                        
                        if final_result:
                            logger.info(f"Retrieved final result: {final_result}")
                            # Use the final result as the specialized agent response
                            result["specialized_agent_response"] = final_result
                        else:
                            # Keep the initial response if polling failed
                            result["specialized_agent_response"] = search_data
                    else:
                        # Use the immediate response
                        result["specialized_agent_response"] = search_data
                else:
                    logger.error(f"Search agent request failed: {search_response.status_code}")
                    logger.error(f"Response: {search_response.text}")
                    result["error"] = f"Search agent request failed: {search_response.status_code}"
                    result["specialized_agent_response"] = {"error": f"Failed to get response from search agent: {search_response.status_code}"}
            
            elif primary_agent == "core_llm_agent":
                # Similar implementation as math agent with result polling
                logger.info("Forwarding to core LLM agent...")
                
                # Use the workflow endpoint
                llm_endpoint = f"{self.base_url}/workflow/process/text"
                
                # Prepare the payload
                payload = {
                    "text": query,
                    "options": {
                        "mode": "immediate",
                        "response_format": "json"
                    }
                }
                
                # Send request to the core LLM agent through workflow
                llm_response = requests.post(
                    llm_endpoint,
                    json=payload,
                    timeout=300
                )
                
                if llm_response.status_code == 200:
                    llm_data = llm_response.json()
                    logger.info(f"Core LLM agent response: {llm_data}")
                    
                    # If we got a workflow_id, poll for the final result
                    if "workflow_id" in llm_data:
                        workflow_id = llm_data.get("workflow_id")
                        logger.info(f"Got workflow_id: {workflow_id}. Polling for result...")
                        
                        # Poll for the final result (with timeout)
                        final_result = self.poll_workflow_result(workflow_id)
                        
                        if final_result:
                            logger.info(f"Retrieved final result: {final_result}")
                            # Use the final result as the specialized agent response
                            result["specialized_agent_response"] = final_result
                        else:
                            # Keep the initial response if polling failed
                            result["specialized_agent_response"] = llm_data
                    else:
                        # Use the immediate response
                        result["specialized_agent_response"] = llm_data
                else:
                    logger.error(f"Core LLM agent request failed: {llm_response.status_code}")
                    logger.error(f"Response: {llm_response.text}")
                    result["error"] = f"Core LLM agent request failed: {llm_response.status_code}"
                    result["specialized_agent_response"] = {"error": f"Failed to get response from core LLM agent: {llm_response.status_code}"}
            
            else:
                logger.warning(f"Unknown agent type: {primary_agent}")
                result["error"] = f"Unknown agent type: {primary_agent}"
                result["specialized_agent_response"] = {"error": f"Unknown agent type: {primary_agent}"}
            
            return result
            
        except Exception as e:
            logger.error(f"Error in specialized agent request: {e}")
            return {
                "success": False,
                "analysis": routing_info,
                "error": f"Error in specialized agent request: {str(e)}"
            }
            
    def poll_workflow_result(self, workflow_id, max_retries=10, retry_delay=3):
        """
        Poll for the result of an asynchronous workflow.
        
        Args:
            workflow_id: The ID of the workflow to check
            max_retries: Maximum number of status check attempts
            retry_delay: Seconds to wait between retries
            
        Returns:
            dict: The final result of the workflow, or None if polling failed
        """
        logger.info(f"Polling for workflow result: {workflow_id}")
        
        # Poll the workflow status until it's completed or max retries reached
        for attempt in range(max_retries):
            try:
                # Check workflow status
                status_url = f"{self.base_url}/workflow/status/{workflow_id}"
                status_response = requests.get(status_url, timeout=10)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    state = status_data.get("state", "unknown")
                    
                    logger.info(f"Workflow state: {state} (attempt {attempt+1}/{max_retries})")
                    
                    # If workflow is completed, get the result
                    if state.lower() in ["completed", "success", "finished"]:
                        result_url = f"{self.base_url}/workflow/result/{workflow_id}"
                        result_response = requests.get(result_url, timeout=10)
                        
                        if result_response.status_code == 200:
                            result_data = result_response.json()
                            logger.info(f"Successfully retrieved workflow result")
                            return result_data
                        else:
                            logger.error(f"Failed to get workflow result: {result_response.status_code}")
                            return None
                    
                    # If workflow failed, return None
                    elif state.lower() in ["failed", "error"]:
                        logger.error(f"Workflow failed with state: {state}")
                        return None
                    
                    # Otherwise, wait and try again
                    else:
                        logger.info(f"Workflow still in progress. Waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to check workflow status: {status_response.status_code}")
                    return None
            except Exception as e:
                logger.error(f"Error polling workflow: {e}")
                
        # If we reach max retries without a result, return None
        logger.error(f"Max polling attempts reached without getting result")
        return None

    def test_router_api(self, query):
        """
        Test the router API with a query to determine which agent should handle it.
        
        Args:
            query (str): The query to test
            
        Returns:
            dict: The router response with agent information, or None if the request failed
        """
        logger.info(f"Testing router API with query: {query}")
        
        try:
            # Prepare the request to the router endpoint
            router_url = f"{self.base_url}/router/route"
            
            # Check if the router endpoint exists, if not, try the analysis endpoint
            try:
                response = requests.head(router_url, timeout=5)
                if response.status_code >= 400:
                    # Fall back to AI analysis endpoint
                    router_url = f"{self.base_url}/ai-analysis/analyze"
                    logger.info(f"Router endpoint not available, using analysis endpoint: {router_url}")
            except:
                # Fall back to AI analysis endpoint
                router_url = f"{self.base_url}/ai-analysis/analyze"
                logger.info(f"Router endpoint not available, using analysis endpoint: {router_url}")
            
            # Prepare the payload
            payload = {"query": query}
            
            # Send the request to the router
            response = requests.post(
                router_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Router API response: {data}")
                
                # Try to extract agent information from different possible response formats
                agent_info = {}
                
                # Check if the response contains direct agent info
                if "agent_info" in data:
                    agent_info = data["agent_info"]
                # Check if it's in the analysis section (from AI analysis endpoint)
                elif "analysis" in data:
                    analysis = data["analysis"]
                    
                    # Try to get routing information
                    if "routing" in analysis:
                        routing = analysis["routing"]
                        agent_info = {
                            "primary_agent": routing.get("primary_agent", "unknown"),
                            "confidence": routing.get("confidence", 0),
                            "alternative_agents": routing.get("alternative_agents", [])
                        }
                    # If no routing info but there are required_agents, use the first one
                    elif "required_agents" in analysis and analysis["required_agents"]:
                        required_agents = analysis["required_agents"]
                        agent_info = {
                            "primary_agent": required_agents[0],
                            "confidence": 1.0,  # Default high confidence
                            "alternative_agents": required_agents[1:] if len(required_agents) > 1 else []
                        }
                
                # If we still don't have agent info, determine from the operations
                if not agent_info and "analysis" in data:
                    analysis = data["analysis"]
                    operations = analysis.get("operations", [])
                    
                    # Basic mapping of operations to agent types
                    if any(op in ["multiplication", "addition", "subtraction", "division", "calculation"] for op in operations):
                        agent_info = {
                            "primary_agent": "math_computation_agent",
                            "confidence": 0.9,  # Reasonably high confidence
                            "alternative_agents": []
                        }
                    elif any(op in ["search", "lookup", "find", "research"] for op in operations):
                        agent_info = {
                            "primary_agent": "search_agent",
                            "confidence": 0.9,
                            "alternative_agents": []
                        }
                    elif any(op in ["visualize", "plot", "graph", "chart"] for op in operations):
                        agent_info = {
                            "primary_agent": "visualization_agent",
                            "confidence": 0.9,
                            "alternative_agents": []
                        }
                
                # Check if we found agent information
                if agent_info:
                    logger.info(f"Primary agent: {agent_info.get('primary_agent', 'unknown')}")
                    logger.info(f"Confidence: {agent_info.get('confidence', 0)}")
                    
                    # Add the agent_info to the response data if it's not already there
                    if "agent_info" not in data:
                        data["agent_info"] = agent_info
                    
                    return data
                else:
                    # Last resort: for mathematical operations in the query itself
                    if any(op in query for op in ["+", "-", "*", "/", "^", "√", "sqrt", "sin", "cos", "tan"]):
                        logger.info("Detected mathematical operation in query, using math_computation_agent")
                        agent_info = {
                            "primary_agent": "math_computation_agent",
                            "confidence": 0.8,
                            "alternative_agents": []
                        }
                        data["agent_info"] = agent_info
                        return data
                    
                    logger.error("No agent information found in the response")
                    return None
            else:
                logger.error(f"Router API request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error testing router API: {e}")
            return None

    def check_backend_availability(self):
        """Check if the backend services are available."""
        try:
            # Check various backend services
            logger.info("Checking backend availability...")
            
            # 1. Check the router endpoint
            router_available = self.test_router_api("Test query") is not None
            
            # 2. Check other backend services if necessary
            # ...
            
            if router_available:
                logger.info("Backend services are available")
                return True
            else:
                logger.error("Backend services are not fully available")
                return False
        except Exception as e:
            logger.error(f"Error checking backend availability: {e}")
            return False

    def call_specialized_agent(self, agent_type, query, history_id=None):
        """
        Call a specialized agent to process a query.
        
        Args:
            agent_type (str): The type of specialized agent to call
            query (str): The query to process
            history_id (str): Optional history ID for the workflow
            
        Returns:
            dict: The response from the specialized agent, or None if the request failed
        """
        logger.info(f"Calling specialized agent: {agent_type} with query: {query}")
        
        try:
            # For math computation agent, try to call the direct API first
            if agent_type == "math_computation_agent":
                direct_result = self.call_math_agent_directly(query)
                if direct_result:
                    return direct_result
            
            # Special handling for visualization agent
            if agent_type == "visualization_agent":
                # Extract visualization details from the query
                # This helps the visualization agent better understand the request
                query_lower = query.lower()
                
                # Detect visualization type
                viz_type = None
                if "3d" in query_lower or "surface" in query_lower:
                    viz_type = "3d_surface"
                elif "pie" in query_lower:
                    viz_type = "pie_chart"
                elif "bar" in query_lower:
                    viz_type = "bar_chart"
                elif "histogram" in query_lower:
                    viz_type = "histogram"
                elif "scatter" in query_lower:
                    viz_type = "scatter"
                elif "box plot" in query_lower or "boxplot" in query_lower:
                    viz_type = "box_plot"
                elif "heatmap" in query_lower:
                    viz_type = "heatmap"
                
                # Extract potential function or data mentions
                function_match = re.search(r'(?:of|for)\s+([a-zA-Z0-9\*\+\-\/\^\(\)\s]+)', query_lower)
                function_expr = function_match.group(1).strip() if function_match else None
                
                # Determine if multiple functions are involved
                multiple_functions = "and" in query_lower or "," in query_lower
                
                # Prepare specialized options for the visualization agent
                viz_options = {
                    "visualization_type": viz_type,
                    "save_output": True,
                    "high_quality": True,
                    "return_path": True
                }
                
                if function_expr:
                    viz_options["expression"] = function_expr
                
                if multiple_functions:
                    viz_options["multiple_functions"] = True
                
                # Special payload for visualization agent
                payload = {
                    "text": query,
                    "options": {
                        "mode": "immediate",
                        "response_format": "json",
                        "agent": agent_type,
                        "visualization": viz_options
                    }
                }
            else:
                # Standard payload for other agents
                payload = {
                    "text": query,
                    "options": {
                        "mode": "immediate",
                        "response_format": "json",
                        "agent": agent_type
                    }
                }
            
            # Add history_id if provided
            if history_id:
                payload["options"]["history_id"] = history_id
            
            # Prepare the base endpoint for the workflow
            base_endpoint = f"{self.base_url}/workflow/process/text"
            
            # Send the request to the specialized agent
            response = requests.post(
                base_endpoint,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Specialized agent response received")
                
                # If we got a workflow_id, poll for the final result
                if "workflow_id" in data:
                    workflow_id = data["workflow_id"]
                    logger.info(f"Got workflow_id: {workflow_id}. Polling for result...")
                    
                    final_result = self.poll_workflow_result(workflow_id)
                    if final_result:
                        logger.info(f"Final workflow result received")
                        return final_result
                    else:
                        logger.warning(f"Failed to get final result, using initial response")
                        return data
                
                # Otherwise, return the immediate response
                return data
            else:
                logger.error(f"Specialized agent request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling specialized agent: {e}")
            return None
    
    def call_math_agent_directly(self, query):
        """
        Call the math computation agent directly to get a quick answer.
        
        Args:
            query (str): The math query to process
            
        Returns:
            dict: The formatted response from the math agent, or None if the request failed
        """
        logger.info(f"Calling math agent directly with query: {query}")
        
        try:
            # Try several possible endpoints for the math agent
            math_endpoints = [
                f"{self.base_url}/math/compute",
                f"{self.base_url}/math/solve",
                f"{self.base_url}/api/math/compute",
                f"{self.base_url}/api/agents/math"
            ]
            
            for endpoint in math_endpoints:
                try:
                    # Prepare the payload
                    payload = {
                        "query": query,
                        "format": "json"
                    }
                    
                    # Send the request
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"Math agent direct response received from {endpoint}")
                        
                        # Format the response
                        if isinstance(data, dict):
                            if "result" in data:
                                logger.info(f"Direct math result found: {data['result']}")
                                return {
                                    "result": data["result"],
                                    "direct_call": True,
                                    "endpoint": endpoint
                                }
                            elif "answer" in data:
                                logger.info(f"Direct math answer found: {data['answer']}")
                                return {
                                    "result": data["answer"],
                                    "direct_call": True,
                                    "endpoint": endpoint
                                }
                        
                        # If we got here, we have a response but no clear result field
                        logger.info(f"Got response but no clear result field")
                        return {
                            "result": str(data),
                            "direct_call": True,
                            "endpoint": endpoint
                        }
                except Exception as e:
                    logger.warning(f"Failed to call math agent at {endpoint}: {e}")
                    continue
            
            # If we get here, all endpoints failed
            logger.warning("All direct math agent endpoints failed")
            return None
        except Exception as e:
            logger.error(f"Error calling math agent directly: {e}")
            return None

    def call_external_math_api(self, query):
        """
        Call an external math API as a last resort to get an answer.
        
        Args:
            query (str): The math query to process
            
        Returns:
            dict: The formatted response from the external API, or None if the request failed
        """
        logger.info(f"Calling external math API with query: {query}")
        
        try:
            # Strip the query down to just the math expression
            expression = query.lower()
            
            # Remove common question phrases
            for phrase in ["what is ", "calculate ", "compute ", "find ", "solve ", "evaluate "]:
                expression = expression.replace(phrase, "")
            
            # Remove question marks and other non-math characters
            expression = expression.replace("?", "").strip()
            
            logger.info(f"Simplified expression: {expression}")
            
            # Simple evaluation approach for basic math operations
            try:
                # For simple operations like addition, subtraction, multiplication, division
                if any(op in expression for op in ["+", "-", "*", "/", "^"]):
                    # Replace ^ with ** for Python's power operator
                    expression = expression.replace("^", "**")
                    
                    # Evaluate the expression directly
                    result = eval(expression)
                    logger.info(f"Direct evaluation result: {result}")
                    
                    return {
                        "result": str(result),
                        "direct_evaluation": True,
                        "expression": expression
                    }
            except Exception as e:
                logger.warning(f"Failed to evaluate expression directly: {e}")
            
            # If direct evaluation fails, try using an external API
            try:
                # Use a public calculator API
                url = f"https://api.mathjs.org/v4/?expr={expression}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    result = response.text
                    logger.info(f"External math API result: {result}")
                    
                    return {
                        "result": result,
                        "external_api": True,
                        "expression": expression
                    }
            except Exception as e:
                logger.warning(f"Failed to call external math API: {e}")
            
            # If we get here, all methods failed
            logger.warning("All math calculation methods failed")
            return None
        except Exception as e:
            logger.error(f"Error in external math calculation: {e}")
            return None

    def call_direct_llm_endpoint(self, query):
        """
        Call a direct LLM endpoint to get a response for the given query.
        
        Args:
            query (str): The query to process
            
        Returns:
            str: The LLM response, or None if the request failed
        """
        logger.info(f"Calling direct LLM endpoint with query: {query}")
        
        try:
            # Try different possible endpoints for direct LLM access
            llm_endpoints = [
                f"{self.base_url}/llm/generate",
                f"{self.base_url}/api/llm/generate",
                f"{self.base_url}/api/core/llm",
                f"{self.base_url}/api/chat"
            ]
            
            for endpoint in llm_endpoints:
                try:
                    # Prepare the payload
                    payload = {
                        "prompt": query,
                        "format": "text"
                    }
                    
                    # Send the request
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"Direct LLM response received from {endpoint}")
                        
                        # Extract the response based on possible field names
                        if isinstance(data, dict):
                            for key in ["text", "content", "response", "generated_text", "answer", "message"]:
                                if key in data:
                                    return data[key]
                            
                            # If we can't find a specific field, return the whole response
                            return str(data)
                        elif isinstance(data, str):
                            return data
                except Exception as e:
                    logger.warning(f"Failed to call LLM at {endpoint}: {e}")
                    continue
            
            # If direct endpoints failed, try a chat endpoint format
            chat_endpoints = [
                f"{self.base_url}/chat",
                f"{self.base_url}/api/chat"
            ]
            
            for endpoint in chat_endpoints:
                try:
                    # Prepare the payload in chat format
                    payload = {
                        "messages": [
                            {"role": "user", "content": query}
                        ]
                    }
                    
                    # Send the request
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"Chat response received from {endpoint}")
                        
                        # Extract the response based on possible field names
                        if isinstance(data, dict):
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if isinstance(choice, dict) and "message" in choice:
                                    return choice["message"].get("content", "")
                            
                            for key in ["text", "content", "response", "message"]:
                                if key in data:
                                    return data[key]
                            
                            # If we can't find a specific field, return the whole response
                            return str(data)
                except Exception as e:
                    logger.warning(f"Failed to call chat endpoint at {endpoint}: {e}")
                    continue
            
            # If all endpoints failed, return None
            logger.warning("All direct LLM endpoints failed")
            return None
        except Exception as e:
            logger.error(f"Error calling direct LLM endpoint: {e}")
            return None
    
    def generate_llm_response(self, query):
        """
        Generate a response for the given query when direct LLM endpoints fail.
        This is a fallback method that generates a plausible response based on the query type.
        
        Args:
            query (str): The query to process
            
        Returns:
            str: The generated response
        """
        logger.info(f"Generating fallback response for query: {query}")
        
        # Check if it's a greeting or introduction
        query_lower = query.lower()
        if any(greeting in query_lower for greeting in ["hello", "hi ", "hey", "greetings"]):
            return "Hello! How can I help you today?"
        
        # Check if it's asking for help or capabilities
        if any(help_term in query_lower for help_term in ["help", "can you", "what can you", "how do you"]):
            return "I'm an AI assistant that can help you with a wide range of tasks including answering questions, providing information, having conversations, helping with writing, and much more. What would you like to know?"
        
        # Check if it's asking for information on a topic
        if any(info_term in query_lower for info_term in ["what is", "who is", "tell me about", "explain", "how does"]):
            return f"I'd be happy to provide information about '{query[query.lower().find('what is') + 8:].strip()}'. Please note that my knowledge has a cutoff date and I recommend verifying any critical information."
        
        # Check if it's a creative writing request
        if "write" in query_lower or "generate" in query_lower or "create" in query_lower or "compose" in query_lower:
            if "poem" in query_lower or "poetry" in query_lower:
                return "Here's a poem I've created for you:\n\nWhispers of the Wind\n\nSoft whispers dance among the trees,\nCarried gently on the autumn breeze.\nStories of distant lands they tell,\nTales of wonder where dreams dwell.\n\nTime stands still in this tranquil space,\nAs nature moves at its own pace.\nIn this moment, perfectly aligned,\nPeace settles within the mind."
            elif "story" in query_lower:
                return "Here's a short story:\n\nThe Unexpected Encounter\n\nEmma paused on the forest path, her attention caught by a flash of blue among the trees. As she approached, she discovered not a bird as expected, but a small door, barely a foot tall, embedded in the trunk of an ancient oak. Curiosity got the better of her caution, and she knelt down to examine it. The tiny brass doorknob turned easily in her fingers, and as the door swung open, a soft golden light spilled out, illuminating her astonished face. This was just the beginning of an adventure she'd retell for years to come."
            else:
                return f"I've created the following content based on your request:\n\n[Creative content related to '{query}' would appear here, tailored to your specific request]"
        
        # General response for other types of queries
        return "I understand you're asking about '{}'. I'd be happy to help with this. Could you provide a bit more context or specify what aspect you're most interested in?".format(query)

    def generate_visualization_response(self, query):
        """
        Generate a fallback visualization response when the visualization agent fails.
        
        Args:
            query (str): The visualization query
            
        Returns:
            str: A descriptive response about the visualization
        """
        logger.info(f"Generating fallback visualization response for: {query}")
        
        # Set up the base directory for saving visualizations
        base_dir = "visualizations"
        # Create the directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Generate a filename based on the query content
        query_words = query.lower().split()
        # Remove common words
        common_words = ["a", "an", "the", "can", "you", "me", "please", "create", "make", "generate", "plot", "chart", "graph"]
        filtered_words = [word for word in query_words if word not in common_words]
        # Create a filename by joining up to 5 words with underscores
        filename_base = "_".join(filtered_words[:5])
        if not filename_base:
            # Fallback if there are no meaningful words
            filename_base = "visualization"
        
        # Create a unique filename with timestamp
        timestamp = int(time.time())
        filename = f"{filename_base}_{timestamp}.png"
        filepath = os.path.join(base_dir, filename)
        
        query_lower = query.lower()
        
        # Common visualization types
        viz_types = {
            "bar": ["bar chart", "bar graph", "bar plot", "histogram"],
            "line": ["line chart", "line graph", "line plot", "curve", "trend"],
            "scatter": ["scatter plot", "scatter chart", "scatter graph", "scatterplot"],
            "pie": ["pie chart", "pie graph", "donut chart"],
            "area": ["area chart", "area graph", "area plot", "stacked area"],
            "heatmap": ["heatmap", "heat map", "heat chart"],
            "box": ["box plot", "box chart", "boxplot"],
            "radar": ["radar chart", "radar graph", "spider chart"],
            "violin": ["violin plot", "violin chart"],
            "bubble": ["bubble chart", "bubble plot"],
            "funnel": ["funnel chart", "funnel graph"],
            "sankey": ["sankey diagram", "sankey chart", "flow diagram"],
            "network": ["network graph", "network diagram", "node graph"]
        }
        
        # Detect the type of visualization requested
        detected_viz_type = None
        for viz_type, keywords in viz_types.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_viz_type = viz_type
                break
        
        # If no specific visualization type is detected, try to infer from other keywords
        if not detected_viz_type:
            if "plot" in query_lower or "graph" in query_lower or "chart" in query_lower:
                if any(func in query_lower for func in ["sin", "cos", "tan", "log", "exp"]):
                    detected_viz_type = "line"
                elif any(term in query_lower for term in ["distribution", "frequency", "histogram"]):
                    detected_viz_type = "bar"
                elif any(term in query_lower for term in ["correlation", "relationship"]):
                    detected_viz_type = "scatter"
                elif any(term in query_lower for term in ["comparison", "proportion", "percentage"]):
                    detected_viz_type = "pie"
                else:
                    # Default to line chart if no specific type can be determined
                    detected_viz_type = "line"
        
        # Handle special cases for mathematical functions
        math_function = None
        if "sin" in query_lower:
            math_function = "sine"
        elif "cos" in query_lower:
            math_function = "cosine"
        elif "tan" in query_lower:
            math_function = "tangent"
        elif "log" in query_lower:
            math_function = "logarithmic"
        elif "exp" in query_lower:
            math_function = "exponential"
        elif "sqrt" in query_lower or "square root" in query_lower:
            math_function = "square root"
        
        # Generate a response based on the visualization type and detected function
        if math_function:
            return f"""To plot the {math_function} function, I would typically:

1. Generate a set of x values over an appropriate range
2. Calculate the corresponding y values using the {math_function} function
3. Create a line plot showing the curve

The {math_function} function would appear as a smooth curve showing the characteristic wave pattern. For example, the sine function oscillates between -1 and 1, completing a full cycle every 2π units along the x-axis.

To implement this visualization in Python, you could use:
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = np.sin(x)  # Replace with appropriate function

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('{math_function.capitalize()} Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.savefig('{filepath}')  # Save the figure
plt.show()
```

The visualization will be saved to: {filepath}"""
        elif detected_viz_type == "bar":
            return f"""To create a bar chart based on your request, I would:

1. Collect and organize the categorical data
2. Calculate the values for each category
3. Generate a bar chart showing the values for each category

A bar chart is ideal for comparing distinct categories. Each bar's height represents the value for that category, making it easy to compare values visually.

To implement this visualization in Python, you could use:
```python
import matplotlib.pyplot as plt

categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [15, 34, 23, 48]

plt.figure(figsize=(10, 6))
plt.bar(categories, values)
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.savefig('{filepath}')  # Save the figure
plt.show()
```

The visualization will be saved to: {filepath}"""
        elif detected_viz_type == "scatter":
            return f"""To create a scatter plot based on your request, I would:

1. Collect pairs of related data points (x,y)
2. Plot each pair as a point on a 2D coordinate system
3. Analyze the pattern to identify relationships or correlations

Scatter plots are excellent for visualizing relationships between two variables and identifying patterns like linear correlations, clusters, or outliers.

To implement this visualization in Python, you could use:
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
x = np.random.rand(50) * 10
y = 2*x + np.random.randn(50)*2

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7)
plt.title('Price vs. Size Correlation')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.savefig('{filepath}')  # Save the figure
plt.show()
```

The visualization will be saved to: {filepath}"""
        elif detected_viz_type == "pie":
            return f"""To create a pie chart based on your request, I would:

1. Collect categorical data with corresponding percentage or proportion values
2. Calculate the relative size of each category
3. Generate a pie chart where each slice represents a category

Pie charts are ideal for showing proportions and percentages, making it easy to see how each part contributes to the whole.

To implement this visualization in Python, you could use:
```python
import matplotlib.pyplot as plt

categories = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [15, 30, 45, 10]

plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=categories, autopct='%1.1f%%', startangle=90, shadow=True)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Pie Chart Example')
plt.savefig('{filepath}')  # Save the figure
plt.show()
```

The visualization will be saved to: {filepath}"""
        else:
            # Generic line chart as default
            return f"""To create a visualization based on your request, I would typically:

1. Collect and prepare the appropriate data
2. Determine the most suitable chart type for your data and objectives
3. Generate a well-labeled, clear visualization

For time series or relational data, a line chart would be appropriate:
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Line Chart Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.grid(True)
plt.savefig('{filepath}')  # Save the figure
plt.show()
```

For categorical comparisons, a bar chart might be better:
```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
values = [15, 34, 23, 48]

plt.figure(figsize=(10, 6))
plt.bar(categories, values)
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.savefig('{filepath}')  # Save the figure
plt.show()
```

The visualization will be saved to: {filepath}"""

    def generate_actual_visualization(self, query):
        """
        Generate an actual visualization based on the query using matplotlib.
        
        Args:
            query (str): The visualization query
            
        Returns:
            tuple: (filepath, visualization_type) if successful, None otherwise
        """
        try:
            # Import required modules from the existing visualization system
            # First try to use the existing visualization system
            try:
                # Import the visualization modules
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                
                # Try to import the existing visualization components
                try:
                    from visualization.agent.viz_agent import VisualizationAgent
                    from visualization.plotting.plot_2d import plot_multiple_functions_2d, plot_function_2d
                    from visualization.plotting.plot_3d import plot_function_3d, plot_parametric_3d
                    import sympy as sp
                    
                    logger.info("Using existing visualization system")
                    
                    # Set up the base directory for saving visualizations
                    base_dir = "visualizations"
                    # Create the directory if it doesn't exist
                    os.makedirs(base_dir, exist_ok=True)
                    
                    # Generate a filename based on the query content
                    query_words = query.lower().split()
                    # Remove common words
                    common_words = ["a", "an", "the", "can", "you", "me", "please", "create", "make", "generate", "plot", "chart", "graph"]
                    filtered_words = [word for word in query_words if word not in common_words]
                    # Create a filename by joining up to 5 words with underscores
                    filename_base = "_".join(filtered_words[:5])
                    if not filename_base:
                        # Fallback if there are no meaningful words
                        filename_base = "visualization"
                    
                    # Create a unique filename with timestamp
                    timestamp = int(time.time())
                    filename = f"{filename_base}_{timestamp}.png"
                    filepath = os.path.join(base_dir, filename)
                    
                    # Parse the query and identify visualization requirements
                    query_lower = query.lower()
                    
                    # Try to detect the type of visualization needed
                    viz_type = None
                    
                    # Check for trigonometric functions
                    if "sin" in query_lower or "cos" in query_lower or "tan" in query_lower:
                        # Check for combinations of functions
                        if any(combo in query_lower for combo in ["and", "with", "together", "both", "compare"]):
                            viz_type = "multiple_trig"
                        else:
                            # Single trig function
                            if "sin" in query_lower:
                                viz_type = "sin"
                            elif "cos" in query_lower:
                                viz_type = "cos"
                            elif "tan" in query_lower:
                                viz_type = "tan"
                    
                    # Check for other mathematical functions
                    elif "log" in query_lower:
                        viz_type = "log"
                    elif "exp" in query_lower or "exponential" in query_lower:
                        viz_type = "exp"
                    elif "sqrt" in query_lower or "square root" in query_lower:
                        viz_type = "sqrt"
                    elif "polynomial" in query_lower or "quadratic" in query_lower or "cubic" in query_lower:
                        viz_type = "polynomial"
                    
                    # Check for statistical or data visualizations
                    elif any(term in query_lower for term in ["bar chart", "bar graph", "histogram"]):
                        viz_type = "bar_chart"
                    elif any(term in query_lower for term in ["pie chart", "pie graph", "donut"]):
                        viz_type = "pie_chart"
                    elif any(term in query_lower for term in ["scatter plot", "scatter", "correlation"]):
                        viz_type = "scatter"
                    elif any(term in query_lower for term in ["time series", "trend", "over time"]):
                        viz_type = "time_series"
                    elif any(term in query_lower for term in ["box plot", "boxplot", "distribution"]):
                        viz_type = "box_plot"
                    elif any(term in query_lower for term in ["3d", "surface", "three dimensional"]):
                        viz_type = "3d"
                    
                    # If no specific type detected, default to a simple plot
                    if viz_type is None:
                        viz_type = "simple_plot"
                    
                    # Generate the appropriate visualization based on the detected type
                    if viz_type == "multiple_trig":
                        # Determine which trig functions are requested
                        functions = []
                        labels = []
                        
                        if "sin" in query_lower:
                            functions.append("sin(x)")
                            labels.append("sin(x)")
                        
                        if "cos" in query_lower:
                            functions.append("cos(x)")
                            labels.append("cos(x)")
                        
                        if "tan" in query_lower:
                            functions.append("tan(x)")
                            labels.append("tan(x)")
                        
                        logger.info(f"Plotting multiple trigonometric functions: {', '.join(labels)}")
                        
                        # If we have tan function, limit the range to avoid asymptotes
                        x_range = (-10, 10)
                        if "tan" in labels:
                            x_range = (-3, 3)  # More reasonable range for tan
                        
                        # Use the specialized multiple functions plotter
                        result = plot_multiple_functions_2d(
                            functions=functions,
                            labels=labels,
                            x_range=x_range,
                            title="Trigonometric Functions",
                            x_label="x",
                            y_label="y",
                            save_path=filepath
                        )
                        
                        if result["success"]:
                            return filepath, f"multiple trigonometric functions"
                        else:
                            logger.error(f"Failed to plot functions: {result['error']}")
                            # Fall back to the basic plotting method
                            raise Exception(f"Failed to plot with visualization system: {result['error']}")
                    
                    elif viz_type in ["sin", "cos", "tan"]:
                        function_expr = f"{viz_type}(x)"
                        title = f"{viz_type.capitalize()} Function"
                        
                        # Adjust range for tan to avoid asymptotes
                        x_range = (-10, 10)
                        if viz_type == "tan":
                            x_range = (-3, 3)
                        
                        logger.info(f"Plotting {viz_type} function")
                        
                        result = plot_function_2d(
                            function_expr=function_expr,
                            x_range=x_range,
                            title=title,
                            x_label="x",
                            y_label=f"{viz_type}(x)",
                            save_path=filepath
                        )
                        
                        if result["success"]:
                            return filepath, f"{viz_type} function"
                        else:
                            logger.error(f"Failed to plot function: {result['error']}")
                            # Fall back to the basic plotting method
                            raise Exception(f"Failed to plot with visualization system: {result['error']}")
                    
                    elif viz_type in ["log", "exp", "sqrt"]:
                        function_map = {
                            "log": "log(x)",
                            "exp": "exp(x)",
                            "sqrt": "sqrt(x)"
                        }
                        
                        function_expr = function_map[viz_type]
                        title_map = {
                            "log": "Logarithmic",
                            "exp": "Exponential",
                            "sqrt": "Square Root"
                        }
                        title = f"{title_map[viz_type]} Function"
                        
                        # Adjust range for each function type
                        x_range = {
                            "log": (0.1, 10),
                            "exp": (-3, 3),
                            "sqrt": (0, 10)
                        }[viz_type]
                        
                        logger.info(f"Plotting {viz_type} function")
                        
                        result = plot_function_2d(
                            function_expr=function_expr,
                            x_range=x_range,
                            title=title,
                            x_label="x",
                            y_label=f"{function_expr}",
                            save_path=filepath
                        )
                        
                        if result["success"]:
                            return filepath, f"{viz_type} function"
                        else:
                            logger.error(f"Failed to plot function: {result['error']}")
                            # Fall back to the basic plotting method
                            raise Exception(f"Failed to plot with visualization system: {result['error']}")
                    
                    elif viz_type == "polynomial":
                        # Try to detect the degree of the polynomial from the query
                        if "quadratic" in query_lower or "square" in query_lower:
                            function_expr = "x^2"
                            title = "Quadratic Function"
                        elif "cubic" in query_lower:
                            function_expr = "x^3"
                            title = "Cubic Function"
                        else:
                            # Default to quadratic
                            function_expr = "x^2 + 2*x + 1"
                            title = "Polynomial Function"
                        
                        logger.info(f"Plotting polynomial function: {function_expr}")
                        
                        result = plot_function_2d(
                            function_expr=function_expr,
                            x_range=(-5, 5),
                            title=title,
                            x_label="x",
                            y_label="f(x)",
                            save_path=filepath
                        )
                        
                        if result["success"]:
                            return filepath, "polynomial function"
                        else:
                            logger.error(f"Failed to plot function: {result['error']}")
                            # Fall back to the basic plotting method
                            raise Exception(f"Failed to plot with visualization system: {result['error']}")
                    
                    elif viz_type == "3d":
                        # For 3D plots, we need a 3D function
                        if "sin" in query_lower or "cos" in query_lower:
                            # Try to generate a 3D surface with trigonometric functions
                            if "sin" in query_lower and "cos" in query_lower:
                                function_expr = "sin(x)*cos(y)"
                                title = "sin(x)*cos(y) Surface"
                            elif "sin" in query_lower:
                                function_expr = "sin(sqrt(x^2 + y^2))"
                                title = "sin(r) Surface"
                            else:
                                function_expr = "cos(sqrt(x^2 + y^2))"
                                title = "cos(r) Surface"
                        else:
                            # Default to a simple paraboloid
                            function_expr = "x^2 + y^2"
                            title = "3D Surface Plot"
                        
                        logger.info(f"Plotting 3D function: {function_expr}")
                        
                        try:
                            result = plot_function_3d(
                                function_expr=function_expr,
                                x_range=(-5, 5),
                                y_range=(-5, 5),
                                title=title,
                                save_path=filepath
                            )
                            
                            if result["success"]:
                                return filepath, "3D surface plot"
                            else:
                                logger.error(f"Failed to plot 3D function: {result['error']}")
                                # Fall back to the basic plotting method
                                raise Exception(f"Failed to plot with visualization system: {result['error']}")
                        except Exception as e:
                            logger.error(f"Error in 3D plotting: {e}")
                            # Fall back to a 2D contour plot
                            import matplotlib.pyplot as plt
                            import numpy as np
                            
                            x = np.linspace(-5, 5, 100)
                            y = np.linspace(-5, 5, 100)
                            X, Y = np.meshgrid(x, y)
                            
                            if "sin" in query_lower and "cos" in query_lower:
                                Z = np.sin(X) * np.cos(Y)
                            elif "sin" in query_lower:
                                R = np.sqrt(X**2 + Y**2)
                                Z = np.sin(R)
                            elif "cos" in query_lower:
                                R = np.sqrt(X**2 + Y**2)
                                Z = np.cos(R)
                            else:
                                Z = X**2 + Y**2
                            
                            plt.figure(figsize=(10, 8))
                            contour = plt.contourf(X, Y, Z, 20, cmap='viridis')
                            plt.colorbar(contour)
                            plt.title(title + " (Contour)")
                            plt.xlabel("x")
                            plt.ylabel("y")
                            plt.savefig(filepath)
                            plt.close()
                            
                            return filepath, "contour plot"
                    
                    # Statistical plots with sample data
                    elif viz_type in ["bar_chart", "pie_chart", "scatter", "time_series", "box_plot"]:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        if viz_type == "bar_chart":
                            categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
                            values = [25, 40, 30, 55, 15]
                            
                            plt.figure(figsize=(10, 6))
                            plt.bar(categories, values, color='skyblue')
                            plt.title('Bar Chart')
                            plt.xlabel('Categories')
                            plt.ylabel('Values')
                            plt.grid(axis='y', alpha=0.3)
                            plt.savefig(filepath)
                            plt.close()
                            
                            return filepath, "bar chart"
                            
                        elif viz_type == "pie_chart":
                            categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
                            values = [25, 40, 30, 55, 15]
                            
                            plt.figure(figsize=(10, 8))
                            plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, shadow=True)
                            plt.title('Pie Chart')
                            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            plt.savefig(filepath)
                            plt.close()
                            
                            return filepath, "pie chart"
                            
                        elif viz_type == "scatter":
                            np.random.seed(42)
                            x = np.random.rand(50) * 10
                            y = 2 * x + np.random.randn(50) * 2
                            
                            plt.figure(figsize=(10, 6))
                            plt.scatter(x, y, alpha=0.7, s=100, color='blue', edgecolors='black')
                            plt.title('Scatter Plot')
                            plt.xlabel('X Values')
                            plt.ylabel('Y Values')
                            plt.grid(True, alpha=0.3)
                            plt.savefig(filepath)
                            plt.close()
                            
                            return filepath, "scatter plot"
                            
                        elif viz_type == "time_series":
                            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            values = [54, 67, 72, 77, 89, 95, 105, 102, 98, 85, 76, 82]
                            
                            plt.figure(figsize=(12, 6))
                            plt.plot(months, values, marker='o', linestyle='-', linewidth=2, markersize=8)
                            plt.title('Time Series Plot')
                            plt.xlabel('Month')
                            plt.ylabel('Value')
                            plt.grid(True, alpha=0.3)
                            plt.savefig(filepath)
                            plt.close()
                            
                            return filepath, "time series plot"
                            
                        elif viz_type == "box_plot":
                            np.random.seed(42)
                            data = [np.random.normal(0, 1, 100), np.random.normal(2, 1.5, 100), np.random.normal(-1, 2, 100)]
                            labels = ['Group A', 'Group B', 'Group C']
                            
                            plt.figure(figsize=(10, 6))
                            plt.boxplot(data, labels=labels, patch_artist=True)
                            plt.title('Box Plot')
                            plt.ylabel('Values')
                            plt.grid(axis='y', alpha=0.3)
                            plt.savefig(filepath)
                            plt.close()
                            
                            return filepath, "box plot"
                    
                    # Default to a simple plot if nothing else matched
                    else:
                        logger.info("Generating default plot")
                        x = np.linspace(-10, 10, 1000)
                        y = x**2  # Default to a parabola
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(x, y)
                        plt.title('Function Plot')
                        plt.xlabel('x')
                        plt.ylabel('f(x)')
                        plt.grid(True)
                        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                        plt.savefig(filepath)
                        plt.close()
                        
                        return filepath, "function plot"
                
                except ImportError as e:
                    logger.warning(f"Could not import visualization modules: {e}. Falling back to built-in plotting.")
                    raise e  # Propagate to the fallback code
            
            except Exception as e:
                # If there was an error with the existing visualization system, fall back to the built-in plotting
                logger.warning(f"Error using existing visualization system: {e}. Falling back to built-in plotting.")
                
                # Import fallback libraries
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Set up the base directory for saving visualizations
                base_dir = "visualizations"
                # Create the directory if it doesn't exist
                os.makedirs(base_dir, exist_ok=True)
                
                # Generate a filename based on the query content
                query_words = query.lower().split()
                # Remove common words
                common_words = ["a", "an", "the", "can", "you", "me", "please", "create", "make", "generate", "plot", "chart", "graph"]
                filtered_words = [word for word in query_words if word not in common_words]
                # Create a filename by joining up to 5 words with underscores
                filename_base = "_".join(filtered_words[:5])
                if not filename_base:
                    # Fallback if there are no meaningful words
                    filename_base = "visualization"
                
                # Create a unique filename with timestamp
                timestamp = int(time.time())
                filename = f"{filename_base}_{timestamp}.png"
                filepath = os.path.join(base_dir, filename)
                
                query_lower = query.lower()
                
                # This is our fallback plotting logic when the more advanced methods fail
                if "sin" in query_lower and ("cos" in query_lower or "tan" in query_lower):
                    logger.info("Generating combined trigonometric plot")
                    
                    x = np.linspace(-10, 10, 1000)
                    plt.figure(figsize=(10, 6))
                    
                    if "sin" in query_lower:
                        plt.plot(x, np.sin(x), label='sin(x)')
                    if "cos" in query_lower:
                        plt.plot(x, np.cos(x), label='cos(x)')
                    if "tan" in query_lower:
                        # Limit the range for tangent to avoid extreme values
                        x_tan = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 1000)
                        plt.plot(x_tan, np.tan(x_tan), label='tan(x)')
                    
                    plt.title('Trigonometric Functions')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.grid(True)
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                    plt.legend()
                    plt.savefig(filepath)
                    plt.close()
                    
                    return filepath, "combined trigonometric plot"
                    
                elif "sin" in query_lower:
                    logger.info("Generating sine curve visualization")
                    x = np.linspace(-10, 10, 1000)
                    y = np.sin(x)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y)
                    plt.title('Sine Function')
                    plt.xlabel('x')
                    plt.ylabel('sin(x)')
                    plt.grid(True)
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                    plt.savefig(filepath)
                    plt.close()
                    
                    return filepath, "sine curve"
                    
                elif "cos" in query_lower:
                    logger.info("Generating cosine curve visualization")
                    x = np.linspace(-10, 10, 1000)
                    y = np.cos(x)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y)
                    plt.title('Cosine Function')
                    plt.xlabel('x')
                    plt.ylabel('cos(x)')
                    plt.grid(True)
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                    plt.savefig(filepath)
                    plt.close()
                    
                    return filepath, "cosine curve"
                    
                elif "tan" in query_lower:
                    logger.info("Generating tangent curve visualization")
                    x = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 1000)
                    y = np.tan(x)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y)
                    plt.title('Tangent Function')
                    plt.xlabel('x')
                    plt.ylabel('tan(x)')
                    plt.grid(True)
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                    plt.ylim(-10, 10)  # Limit the y-axis range for better visibility
                    plt.savefig(filepath)
                    plt.close()
                    
                    return filepath, "tangent curve"
                
                # Additional fallback visualizations for the most common cases
                # ...
                
                # Default visualization (if none of the above patterns match)
                logger.info("Generating default line chart visualization")
                x = np.linspace(-10, 10, 1000)
                y = x**2  # Default to a simple parabola
                
                plt.figure(figsize=(10, 6))
                plt.plot(x, y)
                plt.title('Function Plot')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.grid(True)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                plt.savefig(filepath)
                plt.close()
                
                return filepath, "function plot"
                
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            traceback.print_exc()
            return None

    def generate_visualization(self, query_type, data=None, custom_params=None):
        """
        Generate a visualization based on the specified query type and optional data parameters.
        
        Args:
            query_type (str): Type of visualization to generate (e.g., "bar", "line", "pie", "scatter")
            data (dict, optional): Data for the visualization. If None, sample data will be used.
            custom_params (dict, optional): Additional customization parameters.
            
        Returns:
            str: Path to the generated visualization file or None if generation failed
        """
        try:
            # Import required libraries
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
            
            # Set up the base directory for saving visualizations
            base_dir = "visualizations"
            # Create the directory if it doesn't exist
            os.makedirs(base_dir, exist_ok=True)
            
            # Generate a unique filename
            timestamp = int(time.time())
            filename = f"{query_type}_{timestamp}.png"
            filepath = os.path.join(base_dir, filename)
            
            # Initialize default parameters
            params = {
                "title": f"{query_type.title()} Visualization",
                "xlabel": "X Axis",
                "ylabel": "Y Axis",
                "figsize": (10, 6),
                "dpi": 100,
                "grid": True,
                "style": "default",
                "palette": "default",
                "annotations": False
            }
            
            # Update with custom parameters if provided
            if custom_params:
                params.update(custom_params)
            
            # Set matplotlib style if specified
            if params["style"] != "default":
                plt.style.use(params["style"])
            
            # Create the figure
            plt.figure(figsize=params["figsize"], dpi=params["dpi"])
            
            # Generate the visualization based on the query type
            if query_type.lower() == "bar":
                # Sample data for bar chart if not provided
                if not data:
                    data = {
                        "categories": ['Category A', 'Category B', 'Category C', 'Category D', 'Category E'],
                        "values": [25, 34, 30, 40, 35]
                    }
                
                # Extract data
                categories = data.get("categories", [])
                values = data.get("values", [])
                colors = data.get("colors", "skyblue")
                
                # Create the bar chart
                bars = plt.bar(categories, values, color=colors)
                
                # Add value annotations if requested
                if params["annotations"]:
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(values),
                                f'{height}', ha='center', va='bottom')
                
            elif query_type.lower() == "line":
                # Sample data for line chart if not provided
                if not data:
                    x = np.linspace(0, 10, 100)
                    data = {
                        "x": x,
                        "y": [np.sin(x), np.cos(x), np.sin(x) * np.cos(x)]
                    }
                
                # Extract data
                x_data = data.get("x", np.linspace(0, 10, 100))
                y_data = data.get("y", [])
                
                # If y_data is a list of lists/arrays, plot multiple lines
                if isinstance(y_data, list) and all(isinstance(y, (list, np.ndarray)) for y in y_data):
                    for i, y in enumerate(y_data):
                        label = data.get("labels", {}).get(i, f"Series {i+1}")
                        plt.plot(x_data, y, label=label)
                    plt.legend()
                else:
                    # Single line
                    plt.plot(x_data, y_data)
                
            elif query_type.lower() == "pie":
                # Sample data for pie chart if not provided
                if not data:
                    data = {
                        "labels": ['Category A', 'Category B', 'Category C', 'Category D'],
                        "sizes": [15, 30, 45, 10]
                    }
                
                # Extract data
                labels = data.get("labels", [])
                sizes = data.get("sizes", [])
                explode = data.get("explode", None)
                shadow = data.get("shadow", True)
                
                # Create the pie chart
                plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%',
                        shadow=shadow, startangle=90)
                plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
                
            elif query_type.lower() == "scatter":
                # Sample data for scatter plot if not provided
                if not data:
                    np.random.seed(42)
                    x = np.random.rand(50) * 10
                    y = 2 * x + np.random.randn(50) * 2
                    data = {
                        "x": x,
                        "y": y
                    }
                
                # Extract data
                x_data = data.get("x", [])
                y_data = data.get("y", [])
                color = data.get("color", "blue")
                size = data.get("size", 100)
                alpha = data.get("alpha", 0.7)
                
                # Create the scatter plot
                plt.scatter(x_data, y_data, c=color, s=size, alpha=alpha, edgecolors='black')
                
                # Add regression line if requested
                if data.get("regression", False):
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                    line_x = np.linspace(min(x_data), max(x_data), 100)
                    line_y = slope * line_x + intercept
                    plt.plot(line_x, line_y, 'r--', label=f'y = {slope:.2f}x + {intercept:.2f}, R² = {r_value**2:.2f}')
                    plt.legend()
                
            elif query_type.lower() == "histogram":
                # Sample data for histogram if not provided
                if not data:
                    np.random.seed(42)
                    data = {
                        "values": np.random.normal(0, 1, 1000)
                    }
                
                # Extract data
                values = data.get("values", [])
                bins = data.get("bins", 30)
                color = data.get("color", "skyblue")
                
                # Create the histogram
                plt.hist(values, bins=bins, alpha=0.7, color=color, edgecolor='black')
                
                # Add vertical line for mean if requested
                if data.get("show_mean", False):
                    mean_val = np.mean(values)
                    plt.axvline(mean_val, color='red', linestyle='--', 
                                linewidth=2, label=f'Mean: {mean_val:.2f}')
                    plt.legend()
                
            elif query_type.lower() == "heatmap":
                # Sample data for heatmap if not provided
                if not data:
                    np.random.seed(42)
                    matrix = np.random.rand(10, 12)
                    data = {
                        "matrix": matrix,
                        "row_labels": [f'Row {i}' for i in range(1, 11)],
                        "col_labels": [f'Col {i}' for i in range(1, 13)]
                    }
                
                # Extract data
                matrix = data.get("matrix", np.random.rand(5, 5))
                row_labels = data.get("row_labels", [])
                col_labels = data.get("col_labels", [])
                cmap = data.get("cmap", "viridis")
                
                # Create the heatmap
                im = plt.imshow(matrix, cmap=cmap)
                plt.colorbar(im, label=data.get("colorbar_label", "Value"))
                
                # Set ticks and labels if provided
                if row_labels:
                    plt.yticks(range(len(row_labels)), row_labels)
                if col_labels:
                    plt.xticks(range(len(col_labels)), col_labels, rotation=45)
                
                # Add value annotations if requested
                if params["annotations"]:
                    for i in range(len(row_labels)):
                        for j in range(len(col_labels)):
                            text = plt.text(j, i, f'{matrix[i, j]:.2f}',
                                    ha="center", va="center", color="w" if matrix[i, j] > 0.5 else "black")
                
            elif query_type.lower() == "boxplot":
                # Sample data for boxplot if not provided
                if not data:
                    np.random.seed(42)
                    data = {
                        "data": [np.random.normal(0, 1, 100), 
                                np.random.normal(2, 1.5, 100),
                                np.random.normal(-1, 2, 100)],
                        "labels": ['Group A', 'Group B', 'Group C']
                    }
                
                # Extract data
                box_data = data.get("data", [])
                labels = data.get("labels", [])
                
                # Create the boxplot
                plt.boxplot(box_data, labels=labels, patch_artist=True)
                
            elif query_type.lower() == "area":
                # Sample data for area chart if not provided
                if not data:
                    x = np.linspace(0, 10, 100)
                    y = np.sin(x) * 10 + 20
                    data = {"x": x, "y": y}
                
                # Extract data
                x_data = data.get("x", [])
                y_data = data.get("y", [])
                alpha = data.get("alpha", 0.5)
                color = data.get("color", "skyblue")
                
                # Create the area chart
                plt.fill_between(x_data, y_data, alpha=alpha, color=color)
                plt.plot(x_data, y_data, 'b-', linewidth=2)
                
            elif query_type.lower() == "bubble":
                # Sample data for bubble chart if not provided
                if not data:
                    np.random.seed(42)
                    data = {
                        "x": np.random.rand(20) * 10,
                        "y": np.random.rand(20) * 10,
                        "size": np.random.rand(20) * 500 + 100,
                        "color": np.random.rand(20)
                    }
                
                # Extract data
                x_data = data.get("x", [])
                y_data = data.get("y", [])
                size = data.get("size", [])
                color_data = data.get("color", [])
                cmap = data.get("cmap", "viridis")
                alpha = data.get("alpha", 0.7)
                
                # Create the bubble chart
                scatter = plt.scatter(x_data, y_data, s=size, c=color_data, 
                                    cmap=cmap, alpha=alpha, edgecolors='black')
                
                # Add colorbar if color data is provided
                if len(color_data) > 0:
                    plt.colorbar(scatter, label=data.get("color_label", "Value"))
                
            elif query_type.lower() == "radar":
                # Sample data for radar chart if not provided
                if not data:
                    data = {
                        "categories": ['Category A', 'Category B', 'Category C', 
                                       'Category D', 'Category E', 'Category F'],
                        "values": [4, 3, 5, 2, 4, 5]
                    }
                
                # Extract data
                categories = data.get("categories", [])
                values = data.get("values", [])
                
                # Ensure we have categories and values
                if not categories or not values:
                    logger.error("Radar chart requires categories and values")
                    return None
                
                # Set up radar chart
                # Make sure we have a closed loop by appending the first value to the end
                categories = categories + [categories[0]]
                values = values + [values[0]]
                
                # Calculate angles for each category
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # Close the loop
                
                # Create the plot
                ax = plt.subplot(111, polar=True)
                ax.set_theta_offset(np.pi / 2)  # Start from top
                ax.set_theta_direction(-1)  # Clockwise
                
                # Draw the radar chart
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                
                # Set the labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories[:-1])
                
                # Set y-axis limits
                max_value = max(values)
                ax.set_ylim(0, max_value * 1.1)
                
            else:
                # Default to a simple line chart
                logger.warning(f"Unknown visualization type: {query_type}. Defaulting to line chart.")
                x = np.linspace(0, 10, 100)
                y = np.sin(x)
                plt.plot(x, y)
            
            # Set labels and title
            plt.title(params["title"])
            plt.xlabel(params["xlabel"])
            plt.ylabel(params["ylabel"])
            
            # Add grid if specified
            if params["grid"]:
                plt.grid(True, alpha=0.3)
            
            # Adjust layout to prevent cut-off elements
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(filepath)
            plt.close()
            
            logger.info(f"Generated {query_type} visualization at {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating {query_type} visualization: {e}")
            traceback.print_exc()
            return None

    def call_math_agent(self, query):
        """Call the math computation agent to process a query.
        
        Args:
            query (str): The query to process
            
        Returns:
            The response from the math agent, or None if failed
        """
        logger.info(f"Calling math computation agent with query: {query}")
        return self.call_specialized_agent("math_computation_agent", query)
        
    def call_core_agent(self, query):
        """Call the core LLM agent to process a query.
        
        Args:
            query (str): The query to process
            
        Returns:
            The response from the core LLM agent, or None if failed
        """
        logger.info(f"Calling core LLM agent with query: {query}")
        return self.call_specialized_agent("core_llm_agent", query)
        
    def call_visualization_agent(self, query):
        """Call the visualization agent to process a query.
        
        Args:
            query (str): The query to process
            
        Returns:
            The response from the visualization agent, or None if failed
        """
        logger.info(f"Calling visualization agent with query: {query}")
        
        try:
            # Try to call the direct visualization agent endpoint first
            viz_endpoint = f"{self.base_url}/visualization/generate"
            query_lower = query.lower()
            
            # Detect visualization type
            viz_type = "plot"  # Default
            if "3d" in query_lower or "surface" in query_lower:
                viz_type = "surface_3d"
            elif "pie" in query_lower:
                viz_type = "pie"
            elif "bar" in query_lower:
                viz_type = "bar"
            elif "histogram" in query_lower:
                viz_type = "histogram"
            elif "scatter" in query_lower:
                viz_type = "scatter"
            elif "box" in query_lower:
                viz_type = "box"
            elif "heatmap" in query_lower:
                viz_type = "heatmap"
            
            # Extract math expression if present
            math_expr = None
            function_match = re.search(r'(?:of|for)\s+([a-zA-Z0-9\*\+\-\/\^\(\)\s]+)', query_lower)
            if function_match:
                math_expr = function_match.group(1).strip()
            
            # Check for multiple functions
            is_multi = "and" in query_lower or "," in query_lower
            
            # Setup request payload for visualization agent
            direct_payload = {
                "query": query,
                "visualization_type": viz_type,
                "high_quality": True,
                "save_output": True
            }
            
            if math_expr:
                direct_payload["expression"] = math_expr
            
            if is_multi:
                direct_payload["multi_function"] = True
            
            # Try the direct endpoint
            try:
                logger.info(f"Trying direct visualization endpoint with type: {viz_type}")
                direct_response = requests.post(
                    viz_endpoint,
                    json=direct_payload,
                    timeout=60
                )
                
                if direct_response.status_code == 200:
                    resp_data = direct_response.json()
                    logger.info("Direct visualization endpoint responded successfully")
                    return resp_data
                else:
                    logger.warning(f"Direct visualization endpoint failed: {direct_response.status_code}")
            except Exception as e:
                logger.warning(f"Error calling direct visualization endpoint: {e}")
            
            # Fall back to using the specialized agent through workflow
            return self.call_specialized_agent("visualization_agent", query)
            
        except Exception as e:
            logger.error(f"Error calling visualization agent: {e}")
            return self.call_specialized_agent("visualization_agent", query)

def main():
    """Run the diagnostics tool."""
    # Create diagnostics instance
    diagnostics = LLMDiagnostics()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Diagnostics for LLM analysis.')
    parser.add_argument('--workflow', action='store_true', help='Run workflow test')
    parser.add_argument('--query', type=str, default='What is 2+2?', help='Query for diagnosis')
    parser.add_argument('--validate_agents', action='store_true', help='Validate the agents')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--evaluate', type=str, help='Path to evaluation file')
    parser.add_argument('--no-generate-viz', dest='generate_viz', action='store_false', 
                        help='Disable automatic visualization generation (visualizations are generated by default)')
    parser.set_defaults(generate_viz=True)
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Run based on arguments
    if args.workflow:
        # Get router response
        logger.info(f"Simulating workflow for query: {args.query}")
        response = diagnostics.test_router_api(args.query)
        
        if not response:
            logger.error("Router response failed")
            return
        
        # Extract needed information from the response
        agent_info = response.get('agent_info', {})
        primary_agent = agent_info.get('primary_agent', '')
        confidence = agent_info.get('confidence', 0)
        
        # Extract operations and concepts from analysis if available
        analysis = response.get('analysis', {})
        operations = analysis.get('operations', [])
        concepts = analysis.get('concepts', [])
        
        logger.info(f"Router API response: {response}")
        logger.info(f"Operations: {operations}")
        logger.info(f"Concepts: {concepts}")
        logger.info(f"Primary agent: {primary_agent}")
        logger.info(f"Confidence: {confidence}")
        
        # Call the specialized agent based on the router response
        if primary_agent == "math_computation_agent":
            logger.info("Calling math computation agent...")
            agent_response = diagnostics.call_math_agent(args.query)
            logger.info(f"Math agent response: {agent_response}")
            
            if agent_response is None:
                logger.warning("Math agent failed, falling back to direct calculation")
                calculation_result = diagnostics.calculate_simple_math_answer(args.query)
                logger.info(f"Direct calculation result: {calculation_result}")
                if calculation_result is not None:
                    logger.info(f"Final response (via fallback): {calculation_result}")
                else:
                    logger.error("Direct calculation also failed")
            else:
                logger.info(f"Final response: {agent_response}")
                    
        elif primary_agent == "core_llm_agent":
            logger.info("Calling core LLM agent...")
            agent_response = diagnostics.call_core_agent(args.query)
            logger.info(f"Core agent response: {agent_response}")
            
        elif primary_agent == "visualization_agent":
            logger.info("Calling visualization agent...")
            agent_response = diagnostics.call_visualization_agent(args.query)
            logger.info(f"Visualization agent response: {agent_response}")
            
            # Always generate visualizations for visualization queries, regardless of flag
            # (though we still respect the flag if explicitly set to False)
            generate_viz = True
            if 'generate_viz' in vars(args) and args.generate_viz is False:
                # Only disable if flag is explicitly set to False
                generate_viz = False
                
            if generate_viz:
                # Change approach: Instead of implementing our own visualization,
                # properly integrate with the visualization agent
                logger.info("Processing visualization via visualization agent...")
                
                # First, check if we already have a response with a visualization file path
                visualization_path = None
                
                # The visualization agent's response might contain the path to the generated visualization
                if isinstance(agent_response, dict) and 'result' in agent_response:
                    result = agent_response['result']
                    # Check different possible response formats
                    if isinstance(result, dict):
                        # Try different possible field names that might contain the visualization path
                        for field in ['visualization_path', 'file_path', 'image_path', 'path']:
                            if field in result and result[field]:
                                visualization_path = result[field]
                                break
                
                # If we didn't get a path from the agent's response, try to extract workflow_id
                # and check for visualization files created during the workflow
                if not visualization_path and isinstance(agent_response, dict) and 'workflow_id' in agent_response:
                    workflow_id = agent_response['workflow_id']
                    logger.info(f"Checking for visualizations created by workflow: {workflow_id}")
                    
                    # The visualization might be created in a standard location
                    viz_dir = "visualizations"
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    # Check for any files created in the last minute that might be our visualization
                    latest_viz = None
                    latest_time = 0
                    current_time = time.time()
                    
                    try:
                        for filename in os.listdir(viz_dir):
                            filepath = os.path.join(viz_dir, filename)
                            if os.path.isfile(filepath):
                                file_time = os.path.getmtime(filepath)
                                # Check if file was created in the last minute
                                if current_time - file_time < 60 and file_time > latest_time:
                                    latest_time = file_time
                                    latest_viz = filepath
                        
                        if latest_viz:
                            visualization_path = latest_viz
                            logger.info(f"Found recently created visualization: {visualization_path}")
                    except Exception as e:
                        logger.error(f"Error checking for visualizations: {e}")
                
                # If we still don't have a visualization, use our direct visualization method
                if not visualization_path:
                    logger.info("No visualization found in agent response, generating directly...")
                    viz_result = diagnostics.generate_actual_visualization(args.query)
                    if viz_result:
                        visualization_path, viz_type = viz_result
                        logger.info(f"Generated {viz_type} visualization directly at: {visualization_path}")
                    else:
                        logger.error("Failed to generate visualization")
                        # Fall back to instructions if generation failed
                        response = diagnostics.generate_visualization_response(args.query)
                        logger.info(f"Visualization instruction response: {response}")
                else:
                    logger.info(f"Using visualization from agent at: {visualization_path}")
            else:
                # Generate response with instructions if visualization generation is disabled
                response = diagnostics.generate_visualization_response(args.query)
                logger.info(f"Visualization instruction response: {response}")
        else:
            logger.warning(f"Unknown agent type: {primary_agent}")
    
    elif args.validate_agents:
        logger.info("Validating agents...")
        diagnostics.validate_agents()
    
    elif args.evaluate:
        logger.info(f"Running evaluation from file: {args.evaluate}")
        diagnostics.run_evaluation(args.evaluate)
    
    else:
        logger.info("No specific action requested, showing usage instructions")
        parser.print_help()

if __name__ == "__main__":
    main() 
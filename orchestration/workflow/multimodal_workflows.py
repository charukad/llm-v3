"""
Workflow definitions for multimodal processing.

This module defines workflow templates for processing multimodal inputs
and coordinating the various agents involved.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..agents.registry import AgentRegistry
from ..message_bus.message_formats import create_message

logger = logging.getLogger(__name__)

class MultimodalWorkflows:
    """Workflow definitions for multimodal processing."""
    
    def __init__(self, registry: AgentRegistry):
        """
        Initialize multimodal workflows.
        
        Args:
            registry: Agent registry for agent lookup
        """
        self.registry = registry
        logger.info("Initialized multimodal workflows")
    
    def get_workflow_definition(self, workflow_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a workflow definition by type.
        
        Args:
            workflow_type: Workflow type
            
        Returns:
            Workflow definition or None if not found
        """
        if workflow_type == "multimodal_processing":
            return self.define_multimodal_processing_workflow()
        elif workflow_type == "cross_modal_reference_resolution":
            return self.define_cross_modal_reference_workflow()
        elif workflow_type == "ambiguity_resolution":
            return self.define_ambiguity_resolution_workflow()
        else:
            logger.warning(f"Unknown workflow type: {workflow_type}")
            return None
    
    def define_multimodal_processing_workflow(self) -> Dict[str, Any]:
        """
        Define workflow for multimodal input processing.
        
        Returns:
            Workflow definition
        """
        return {
            "name": "Multimodal Processing Workflow",
            "description": "Process multimodal input and generate integrated response",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "initial_processing",
                    "description": "Initial processing of input",
                    "agent_type": None,  # Already processed by API
                    "agent_capability": None,
                    "is_complete": True,  # Skip this step since it's done by API
                    "next_steps": ["content_routing"]
                },
                {
                    "name": "content_routing",
                    "description": "Route content to appropriate agents",
                    "agent_type": None,  # Already processed by API
                    "agent_capability": None,
                    "is_complete": True,  # Skip this step since it's done by API
                    "next_steps": ["agent_processing"]
                },
                {
                    "name": "agent_processing",
                    "description": "Process content with appropriate agents",
                    "agent_type": "dynamic",  # Determined by routing
                    "agent_capability": "dynamic",
                    "is_complete": False,
                    "next_steps": ["cross_references"]
                },
                {
                    "name": "cross_references",
                    "description": "Resolve cross-references between modalities",
                    "agent_type": "service",
                    "agent_capability": "reference_resolver",
                    "service_id": "reference_resolver",
                    "is_complete": False,
                    "next_steps": ["context_integration"]
                },
                {
                    "name": "context_integration",
                    "description": "Integrate results into context",
                    "agent_type": "service",
                    "agent_capability": "context_manager",
                    "service_id": "context_manager",
                    "is_complete": False,
                    "next_steps": ["response_generation"]
                },
                {
                    "name": "response_generation",
                    "description": "Generate integrated response",
                    "agent_type": "core_llm",
                    "agent_capability": "generate_response",
                    "agent_id": "core_llm_agent",
                    "is_complete": False,
                    "next_steps": []
                }
            ],
            "error_handling": {
                "default_action": "retry",
                "max_retries": 3,
                "retry_delay": 1.0,  # seconds
                "failure_action": "notify"
            }
        }
    
    def define_cross_modal_reference_workflow(self) -> Dict[str, Any]:
        """
        Define workflow for cross-modal reference resolution.
        
        Returns:
            Workflow definition
        """
        return {
            "name": "Cross-Modal Reference Resolution Workflow",
            "description": "Resolve references between different modalities",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "extract_references",
                    "description": "Extract references from text",
                    "agent_type": "service",
                    "agent_capability": "reference_resolver",
                    "service_id": "reference_resolver",
                    "is_complete": False,
                    "next_steps": ["find_matching_entities"]
                },
                {
                    "name": "find_matching_entities",
                    "description": "Find matching entities in context",
                    "agent_type": "service",
                    "agent_capability": "context_manager",
                    "service_id": "context_manager",
                    "is_complete": False,
                    "next_steps": ["create_references"]
                },
                {
                    "name": "create_references",
                    "description": "Create references between entities",
                    "agent_type": "service",
                    "agent_capability": "reference_resolver",
                    "service_id": "reference_resolver",
                    "is_complete": False,
                    "next_steps": ["update_context"]
                },
                {
                    "name": "update_context",
                    "description": "Update context with new references",
                    "agent_type": "service",
                    "agent_capability": "context_manager",
                    "service_id": "context_manager",
                    "is_complete": False,
                    "next_steps": []
                }
            ],
            "error_handling": {
                "default_action": "continue",
                "max_retries": 1,
                "retry_delay": 0.5,  # seconds
                "failure_action": "log"
            }
        }
    
    def define_ambiguity_resolution_workflow(self) -> Dict[str, Any]:
        """
        Define workflow for ambiguity resolution.
        
        Returns:
            Workflow definition
        """
        return {
            "name": "Ambiguity Resolution Workflow",
            "description": "Handle ambiguous input and user clarification",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "detect_ambiguities",
                    "description": "Detect ambiguities in processed input",
                    "agent_type": "service",
                    "agent_capability": "ambiguity_handler",
                    "service_id": "ambiguity_handler",
                    "is_complete": False,
                    "next_steps": ["generate_clarification_request"]
                },
                {
                    "name": "generate_clarification_request",
                    "description": "Generate clarification request for user",
                    "agent_type": "service",
                    "agent_capability": "ambiguity_handler",
                    "service_id": "ambiguity_handler",
                    "is_complete": False,
                    "next_steps": ["wait_for_clarification"]
                },
                {
                    "name": "wait_for_clarification",
                    "description": "Wait for user clarification",
                    "agent_type": None,  # External input
                    "agent_capability": None,
                    "is_complete": False,
                    "next_steps": ["process_clarification"]
                },
                {
                    "name": "process_clarification",
                    "description": "Process user clarification",
                    "agent_type": "service",
                    "agent_capability": "ambiguity_handler",
                    "service_id": "ambiguity_handler",
                    "is_complete": False,
                    "next_steps": ["continue_processing"]
                },
                {
                    "name": "continue_processing",
                    "description": "Continue with regular processing workflow",
                    "agent_type": None,  # Jumps to other workflow
                    "agent_capability": None,
                    "is_complete": False,
                    "next_steps": []
                }
            ],
            "error_handling": {
                "default_action": "retry",
                "max_retries": 2,
                "retry_delay": 0.5,  # seconds
                "failure_action": "escalate"
            }
        }
    
    def execute_workflow_step(self, workflow: Dict[str, Any], step_name: str, 
                            data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a step in the workflow.
        
        Args:
            workflow: Workflow definition
            step_name: Name of the step to execute
            data: Data for the step
            
        Returns:
            Result of the step execution
        """
        # Find the step definition
        step = None
        for s in workflow["steps"]:
            if s["name"] == step_name:
                step = s
                break
        
        if not step:
            return {
                "success": False,
                "error": f"Step not found: {step_name}"
            }
        
        if step["is_complete"]:
            # Skip this step as it's marked complete
            if step["next_steps"]:
                next_step = step["next_steps"][0]
                return self.execute_workflow_step(workflow, next_step, data)
            else:
                return {
                    "success": True,
                    "message": f"Workflow complete at step: {step_name}"
                }
        
        try:
            # Execute the step based on agent type
            if step["agent_type"] == "service":
                service_id = step["service_id"]
                service_info = self.registry.get_service_info(service_id)
                
                if not service_info:
                    return {
                        "success": False,
                        "error": f"Service not found: {service_id}"
                    }
                
                service_instance = service_info["instance"]
                
                # Call the appropriate method based on step name
                if step["name"] == "extract_references":
                    result = service_instance.extract_references(data)
                elif step["name"] == "find_matching_entities":
                    result = service_instance.find_matching_entities(data)
                elif step["name"] == "create_references":
                    result = service_instance.create_references(data)
                elif step["name"] == "update_context":
                    result = service_instance.update_context(data)
                elif step["name"] == "detect_ambiguities":
                    result = service_instance.detect_ambiguities(data)
                elif step["name"] == "generate_clarification_request":
                    result = service_instance.generate_clarification_request(data)
                elif step["name"] == "process_clarification":
                    result = service_instance.process_clarification(data)
                else:
                    # Default to a generic process method
                    result = service_instance.process(data)
                
            elif step["agent_type"] == "core_llm":
                agent_id = step["agent_id"]
                agent_info = self.registry.get_agent_info(agent_id)
                
                if not agent_info:
                    return {
                        "success": False,
                        "error": f"Agent not found: {agent_id}"
                    }
                
                agent_instance = agent_info["instance"]
                
                # Create message for the agent
                message = create_message(
                    sender="orchestration_manager",
                    recipient=agent_id,
                    message_type=step["agent_capability"],
                    body=data
                )
                
                result = agent_instance.process_message(message)
                
            elif step["agent_type"] == "dynamic":
                # In this case, the agent is determined by the routing
                routing = data.get("routing", {})
                agent_type = routing.get("agent_type")
                
                if not agent_type:
                    return {
                        "success": False,
                        "error": "No agent type specified in routing"
                    }
                
                # Find an agent with the required capability
                agents = self.registry.find_agent_by_capability(agent_type)
                
                if not agents:
                    return {
                        "success": False,
                        "error": f"No agent found with capability: {agent_type}"
                    }
                
                agent_id = agents[0]
                agent_info = self.registry.get_agent_info(agent_id)
                
                if not agent_info:
                    return {
                        "success": False,
                        "error": f"Agent not found: {agent_id}"
                    }
                
                agent_instance = agent_info["instance"]
                
                # Create message for the agent
                message = create_message(
                    sender="orchestration_manager",
                    recipient=agent_id,
                    message_type=agent_type,
                    body=data
                )
                
                result = agent_instance.process_message(message)
                
            elif step["agent_type"] is None:
                # This step is handled externally or is a flow control step
                if step["name"] == "continue_processing":
                    # Jump to another workflow
                    original_workflow_type = data.get("original_workflow_type", "multimodal_processing")
                    original_workflow = self.get_workflow_definition(original_workflow_type)
                    
                    if not original_workflow:
                        return {
                            "success": False,
                            "error": f"Original workflow not found: {original_workflow_type}"
                        }
                    
                    # Start from agent_processing step
                    return self.execute_workflow_step(original_workflow, "agent_processing", data)
                
                elif step["name"] == "wait_for_clarification":
                    # This is handled by the API, so we just return a waiting status
                    return {
                        "success": True,
                        "waiting_for_input": True,
                        "message": "Waiting for user clarification"
                    }
                
                else:
                    # For other None agent types, just pass through
                    result = {
                        "success": True,
                        "message": f"Step {step_name} skipped (no agent)"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported agent type: {step['agent_type']}"
                }
            
            # If there are next steps, continue the workflow
            if step["next_steps"] and not result.get("waiting_for_input", False):
                next_step = step["next_steps"][0]
                
                # Update data with the result of this step
                updated_data = data.copy()
                updated_data[step_name + "_result"] = result
                
                return self.execute_workflow_step(workflow, next_step, updated_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing workflow step {step_name}: {str(e)}")
            
            # Handle errors based on the workflow's error handling configuration
            error_handling = workflow.get("error_handling", {})
            default_action = error_handling.get("default_action", "fail")
            
            if default_action == "retry":
                max_retries = error_handling.get("max_retries", 3)
                retry_count = data.get("retry_count", {}).get(step_name, 0)
                
                if retry_count < max_retries:
                    # Increment retry count
                    retry_counts = data.get("retry_count", {})
                    retry_counts[step_name] = retry_count + 1
                    
                    updated_data = data.copy()
                    updated_data["retry_count"] = retry_counts
                    
                    # Retry after delay
                    import time
                    time.sleep(error_handling.get("retry_delay", 1.0))
                    
                    return self.execute_workflow_step(workflow, step_name, updated_data)
            
            if default_action == "continue" and step["next_steps"]:
                # Continue to next step despite error
                next_step = step["next_steps"][0]
                
                # Update data with the error
                updated_data = data.copy()
                updated_data[step_name + "_error"] = str(e)
                
                return self.execute_workflow_step(workflow, next_step, updated_data)
            
            # Default to fail
            return {
                "success": False,
                "error": f"Error in step {step_name}: {str(e)}",
                "failure_action": error_handling.get("failure_action", "log")
            }

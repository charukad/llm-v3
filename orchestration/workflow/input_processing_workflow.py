"""
Input Processing Workflow for handling requests from the InputAgent.

This workflow manages the processing of requests routed through
the central InputAgent including the execution of specialized agents.
"""
from typing import Dict, Any, List, Optional
import logging

from .workflow_registry import WorkflowDefinition, get_workflow_registry

logger = logging.getLogger(__name__)

class InputProcessingWorkflow(WorkflowDefinition):
    """
    Workflow for processing requests from the central InputAgent.
    
    This workflow handles the execution of requests that have been analyzed 
    and routed by the InputAgent, with detailed instructions for specialized agents.
    """
    
    @classmethod
    def get_workflow_type(cls) -> str:
        """Get the workflow type identifier."""
        return "input_processing"
    
    async def get_initial_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the initial steps for the input processing workflow."""
        # Extract necessary information from context
        agent_type = context.get("routing", {}).get("agent_type", "core_llm")
        target_endpoint = context.get("endpoint", "/process")
        agent_instructions = context.get("instructions", {})
        
        # Check if we have all the required information
        if not agent_type:
            raise ValueError("No agent type specified in routing information")
        
        # Create the main processing step for the target agent
        main_step = {
            "type": agent_type,
            "name": f"process_with_{agent_type}",
            "description": f"Process request with {agent_type} agent",
            "agent": f"{agent_type}_agent",  # Assuming agent IDs follow the pattern {type}_agent
            "capability": self._get_primary_capability(agent_type, agent_instructions),
            "parameters": agent_instructions.get("parameters", {}),
            "context_keys": ["input_type", "content"]
        }
        
        # Add step-by-step instructions if available
        if "step_by_step" in agent_instructions:
            main_step["step_by_step"] = agent_instructions["step_by_step"]
        
        # Return the steps
        return [main_step]
    
    async def determine_next_steps(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine the next steps based on completed steps and context."""
        # For now, we just implement a simple linear workflow
        # In a more complex implementation, this could involve feedback loops or additional steps
        
        # Check if we have completed the main processing step
        if not completed_steps:
            return []
        
        last_step = completed_steps[-1]
        step_name = last_step.get("name", "")
        
        # If we've completed the main processing step, we might want to add a response formatting step
        if step_name.startswith("process_with_"):
            agent_type = last_step.get("type")
            result = context.get("result", {})
            
            # For visualization requests, we might need to format the result
            if agent_type == "visualization" and result.get("success", False):
                return [{
                    "type": "query",
                    "name": "format_visualization_response",
                    "description": "Format the visualization response",
                    "agent": "core_llm_agent",
                    "capability": "generate_response",
                    "parameters": {
                        "response_type": "visualization_explanation",
                        "include_code": True,
                        "format": "markdown"
                    },
                    "context_keys": ["result"]
                }]
            
            # For math computation, we might want to generate an explanation
            if agent_type == "math_computation" and result.get("success", False):
                return [{
                    "type": "query",
                    "name": "explain_math_solution",
                    "description": "Generate explanation for the mathematical solution",
                    "agent": "core_llm_agent",
                    "capability": "explain_math",
                    "parameters": {
                        "include_steps": True,
                        "explanation_level": "intermediate",
                        "format": "latex"
                    },
                    "context_keys": ["result"]
                }]
        
        # If we've completed any of the formatting steps, we're done
        if step_name in ["format_visualization_response", "explain_math_solution"]:
            return []
        
        # Default: no further steps
        return []
    
    def _get_primary_capability(self, agent_type: str, instructions: Dict[str, Any]) -> str:
        """
        Get the primary capability needed for the given agent type and instructions.
        
        Args:
            agent_type: The type of agent
            instructions: The agent instructions
            
        Returns:
            The primary capability
        """
        # Map agent types to default capabilities
        default_capabilities = {
            "core_llm": "generate_response",
            "math_computation": "compute",
            "visualization": "generate_visualization",
            "ocr": "recognize_math",
            "search": "external_search",
            "text_processing": "natural_language_understanding"
        }
        
        # Try to extract capability from instructions
        if "capabilities_needed" in instructions:
            capabilities = instructions["capabilities_needed"]
            if capabilities and isinstance(capabilities, list):
                return capabilities[0]
        
        # Fall back to default capability for the agent type
        return default_capabilities.get(agent_type, "process")


# Register the workflow
def register_input_processing_workflow():
    """Register the input processing workflow with the registry."""
    registry = get_workflow_registry()
    registry.register_workflow_class(InputProcessingWorkflow)

# Register when module is imported
register_input_processing_workflow() 
"""
Workflow definitions for visualization operations.

This module defines workflow templates for visualization generation,
which can be used by the Orchestration Manager.
"""

from typing import Dict, Any, List, Optional

# Define visualization workflow for generating a visualization
VISUALIZATION_GENERATION_WORKFLOW = {
    "name": "Visualization Generation Workflow",
    "description": "Workflow for generating mathematical visualizations",
    "steps": [
        {
            "name": "parse_request",
            "agent_type": None,  # Handled by orchestration manager
            "description": "Parse and validate the visualization request",
            "next_steps": ["determine_visualization_type"]
        },
        {
            "name": "determine_visualization_type",
            "agent_type": "visualization",
            "capability": "determine_visualization_type",
            "description": "Determine the appropriate visualization type based on context",
            "next_steps": ["generate_visualization"]
        },
        {
            "name": "generate_visualization",
            "agent_type": "dynamic",  # Will be determined based on visualization type
            "capability": "generate_visualization",
            "description": "Generate the requested visualization",
            "next_steps": ["format_response"]
        },
        {
            "name": "format_response",
            "agent_type": None,  # Handled by orchestration manager
            "description": "Format the response with visualization data",
            "next_steps": []  # End of workflow
        }
    ],
    "error_handling": {
        "default_action": "retry",
        "max_retries": 2,
        "fallback_steps": {
            "determine_visualization_type": "use_default_visualization",
            "generate_visualization": "generate_text_description"
        }
    },
    "input_validation": {
        "required_fields": ["mathematical_context"],
        "optional_fields": ["visualization_type", "parameters", "interaction_id"]
    }
}

# Define workflow for enhancing mathematical response with visualizations
MATH_RESPONSE_VISUALIZATION_WORKFLOW = {
    "name": "Mathematical Response Visualization Workflow",
    "description": "Workflow for enhancing mathematical responses with visualizations",
    "steps": [
        {
            "name": "analyze_math_response",
            "agent_type": None,  # Handled by orchestration manager
            "description": "Analyze the mathematical response for visualization opportunities",
            "next_steps": ["extract_visualization_contexts"]
        },
        {
            "name": "extract_visualization_contexts",
            "agent_type": "core_llm",
            "capability": "extract_visualization_contexts",
            "description": "Extract contexts for possible visualizations from the response",
            "next_steps": ["determine_visualizations"]
        },
        {
            "name": "determine_visualizations",
            "agent_type": "visualization",
            "capability": "determine_visualization_type",
            "description": "Determine appropriate visualizations for each context",
            "next_steps": ["generate_visualizations"]
        },
        {
            "name": "generate_visualizations",
            "agent_type": "dynamic",  # Will be determined based on visualization types
            "capability": "generate_visualization",
            "description": "Generate all requested visualizations",
            "next_steps": ["enhance_response"]
        },
        {
            "name": "enhance_response",
            "agent_type": "core_llm",
            "capability": "enhance_with_visualizations",
            "description": "Enhance the original response with visualization references",
            "next_steps": []  # End of workflow
        }
    ],
    "error_handling": {
        "default_action": "continue",
        "max_retries": 1,
        "fallback_steps": {
            "generate_visualizations": "return_original_response"
        }
    },
    "input_validation": {
        "required_fields": ["mathematical_response", "interaction_id"],
        "optional_fields": ["max_visualizations"]
    }
}

def get_visualization_workflow(workflow_type: str) -> Dict[str, Any]:
    """
    Get a workflow definition by type.
    
    Args:
        workflow_type: Type of workflow to retrieve
        
    Returns:
        Workflow definition dictionary
    """
    workflows = {
        "visualization_generation": VISUALIZATION_GENERATION_WORKFLOW,
        "math_response_visualization": MATH_RESPONSE_VISUALIZATION_WORKFLOW
    }
    
    return workflows.get(workflow_type, VISUALIZATION_GENERATION_WORKFLOW)

def customize_visualization_workflow(
    workflow_type: str,
    customizations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a customized version of a visualization workflow.
    
    Args:
        workflow_type: Type of workflow to customize
        customizations: Dictionary of customizations to apply
        
    Returns:
        Customized workflow definition
    """
    # Get base workflow
    workflow = get_visualization_workflow(workflow_type).copy()
    
    # Apply customizations
    for key, value in customizations.items():
        if key == "steps":
            # For steps, merge by step name
            step_dict = {step["name"]: step for step in workflow["steps"]}
            for custom_step in value:
                step_name = custom_step["name"]
                if step_name in step_dict:
                    # Update existing step
                    step_dict[step_name].update(custom_step)
                else:
                    # Add new step
                    step_dict[step_name] = custom_step
            # Rebuild steps list
            workflow["steps"] = list(step_dict.values())
        else:
            # For other keys, simple update
            if isinstance(value, dict) and key in workflow and isinstance(workflow[key], dict):
                # Merge dictionaries
                workflow[key].update(value)
            else:
                # Replace value
                workflow[key] = value
    
    return workflow

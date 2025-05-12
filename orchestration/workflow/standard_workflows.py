"""
Standard workflow definitions for common mathematical tasks.

This module defines standard workflow definitions for common
mathematical tasks in the Mathematical Multimodal LLM System.
"""
import logging
from typing import Dict, Any, List, Optional

from orchestration.workflow.workflow_definition import WorkflowDefinition, WorkflowStep

logger = logging.getLogger(__name__)

# Mathematical problem solving workflow
math_problem_steps = [
    WorkflowStep(
        id="classify_query",
        required_capability="query_classification",
        message_type="math_query",
        input_keys=["query"],
        output_keys=["domain", "expression", "confidence"],
        description="Classify the mathematical query and extract expressions"
    ),
    WorkflowStep(
        id="compute_result",
        required_capability="math_computation",
        message_type="math_computation",
        input_keys=["expression", "domain"],
        output_keys=["result", "steps", "symbolic_result"],
        description="Compute the mathematical result"
    ),
    WorkflowStep(
        id="generate_explanation",
        required_capability="explanation_generation",
        message_type="math_explanation",
        input_keys=["domain", "expression", "result", "steps"],
        output_keys=["explanation"],
        description="Generate a step-by-step explanation"
    ),
    WorkflowStep(
        id="generate_visualization",
        required_capability="visualization",
        message_type="visualization_request",
        input_keys=["domain", "expression", "result"],
        output_keys=["visualization"],
        description="Generate visualizations if appropriate"
    ),
    WorkflowStep(
        id="format_response",
        required_capability="response_formatting",
        message_type="format_request",
        input_keys=["domain", "expression", "result", "steps", "explanation", "visualization"],
        output_keys=["response"],
        description="Format the final response"
    )
]

math_problem_workflow = WorkflowDefinition(
    id="math_problem_solving",
    name="Mathematical Problem Solving",
    description="Workflow for solving mathematical problems with step-by-step explanations",
    steps=math_problem_steps,
    timeout=120  # 2 minutes timeout
)

# Handwriting recognition workflow
handwriting_steps = [
    WorkflowStep(
        id="preprocess_image",
        required_capability="image_preprocessing",
        message_type="image_preprocessing",
        input_keys=["image_path"],
        output_keys=["preprocessed_image"],
        description="Preprocess the handwritten image"
    ),
    WorkflowStep(
        id="recognize_symbols",
        required_capability="symbol_recognition",
        message_type="symbol_recognition",
        input_keys=["preprocessed_image"],
        output_keys=["symbols", "confidence"],
        description="Recognize mathematical symbols in the image"
    ),
    WorkflowStep(
        id="analyze_structure",
        required_capability="structure_analysis",
        message_type="structure_analysis",
        input_keys=["symbols", "preprocessed_image"],
        output_keys=["structure", "confidence"],
        description="Analyze the structure of the mathematical expression"
    ),
    WorkflowStep(
        id="generate_latex",
        required_capability="latex_generation",
        message_type="latex_generation",
        input_keys=["structure", "symbols"],
        output_keys=["latex", "confidence"],
        description="Generate LaTeX representation of the expression"
    ),
    WorkflowStep(
        id="process_expression",
        required_capability="math_computation",
        message_type="math_computation",
        input_keys=["latex"],
        output_keys=["result", "steps"],
        description="Process the recognized expression"
    ),
    WorkflowStep(
        id="format_response",
        required_capability="response_formatting",
        message_type="format_request",
        input_keys=["latex", "result", "steps", "confidence"],
        output_keys=["response"],
        description="Format the final response"
    )
]

handwriting_workflow = WorkflowDefinition(
    id="handwriting_recognition",
    name="Handwriting Recognition",
    description="Workflow for recognizing and processing handwritten mathematical expressions",
    steps=handwriting_steps,
    timeout=180  # 3 minutes timeout
)

# Search-enhanced computation workflow
search_steps = [
    WorkflowStep(
        id="classify_query",
        required_capability="query_classification",
        message_type="math_query",
        input_keys=["query"],
        output_keys=["domain", "expression", "confidence"],
        description="Classify the mathematical query and extract expressions"
    ),
    WorkflowStep(
        id="search_information",
        required_capability="search",
        message_type="search_request",
        input_keys=["query", "domain"],
        output_keys=["search_results"],
        description="Search for relevant mathematical information"
    ),
    WorkflowStep(
        id="compute_with_context",
        required_capability="math_computation",
        message_type="math_computation",
        input_keys=["expression", "domain", "search_results"],
        output_keys=["result", "steps", "symbolic_result"],
        description="Compute the result with search context"
    ),
    WorkflowStep(
        id="generate_explanation",
        required_capability="explanation_generation",
        message_type="math_explanation",
        input_keys=["domain", "expression", "result", "steps", "search_results"],
        output_keys=["explanation", "citations"],
        description="Generate explanation with citations"
    ),
    WorkflowStep(
        id="generate_visualization",
        required_capability="visualization",
        message_type="visualization_request",
        input_keys=["domain", "expression", "result"],
        output_keys=["visualization"],
        description="Generate visualizations if appropriate"
    ),
    WorkflowStep(
        id="format_response",
        required_capability="response_formatting",
        message_type="format_request",
        input_keys=["domain", "expression", "result", "steps", "explanation", "visualization", "citations"],
        output_keys=["response"],
        description="Format the final response with citations"
    )
]

search_workflow = WorkflowDefinition(
    id="search_enhanced_computation",
    name="Search-Enhanced Computation",
    description="Workflow for mathematical computation enhanced with external information",
    steps=search_steps,
    timeout=240  # 4 minutes timeout
)

# Visualization-focused workflow
visualization_steps = [
    WorkflowStep(
        id="parse_visualization_request",
        required_capability="query_classification",
        message_type="visualization_parsing",
        input_keys=["query"],
        output_keys=["visualization_type", "expression", "parameters"],
        description="Parse visualization request"
    ),
    WorkflowStep(
        id="prepare_data",
        required_capability="math_computation",
        message_type="data_preparation",
        input_keys=["expression", "visualization_type", "parameters"],
        output_keys=["plot_data", "computed_parameters"],
        description="Prepare data for visualization"
    ),
    WorkflowStep(
        id="generate_visualization",
        required_capability="visualization",
        message_type="visualization_request",
        input_keys=["visualization_type", "plot_data", "computed_parameters"],
        output_keys=["visualization", "image_path"],
        description="Generate the visualization"
    ),
    WorkflowStep(
        id="generate_description",
        required_capability="explanation_generation",
        message_type="description_request",
        input_keys=["visualization_type", "expression", "plot_data", "computed_parameters"],
        output_keys=["description"],
        description="Generate description of the visualization"
    ),
    WorkflowStep(
        id="format_response",
        required_capability="response_formatting",
        message_type="format_request",
        input_keys=["visualization", "image_path", "description", "expression"],
        output_keys=["response"],
        description="Format the final response"
    )
]

visualization_workflow = WorkflowDefinition(
    id="visualization",
    name="Mathematical Visualization",
    description="Workflow for generating mathematical visualizations",
    steps=visualization_steps,
    timeout=120  # 2 minutes timeout
)

# Register all standard workflows
standard_workflows: Dict[str, WorkflowDefinition] = {
    "math_problem_solving": math_problem_workflow,
    "handwriting_recognition": handwriting_workflow,
    "search_enhanced_computation": search_workflow,
    "visualization": visualization_workflow
}

def get_workflow_definition(workflow_type: str) -> Optional[WorkflowDefinition]:
    """
    Get a workflow definition by type.
    
    Args:
        workflow_type: Workflow type
        
    Returns:
        Workflow definition or None if not found
    """
    return standard_workflows.get(workflow_type)

def validate_all_workflows() -> Dict[str, bool]:
    """
    Validate all standard workflows.
    
    Returns:
        Dictionary mapping workflow types to validation results
    """
    results = {}
    
    for workflow_type, workflow in standard_workflows.items():
        valid = workflow.validate()
        results[workflow_type] = valid
        
        if not valid:
            logger.error(f"Workflow {workflow_type} failed validation")
        else:
            logger.info(f"Workflow {workflow_type} passed validation")
    
    return results

def register_custom_workflow(workflow: WorkflowDefinition) -> bool:
    """
    Register a custom workflow.
    
    Args:
        workflow: Workflow definition
        
    Returns:
        True if registration was successful, False otherwise
    """
    # Validate the workflow
    if not workflow.validate():
        logger.error(f"Custom workflow {workflow.id} failed validation")
        return False
    
    # Register the workflow
    standard_workflows[workflow.id] = workflow
    logger.info(f"Registered custom workflow: {workflow.id}")
    
    return True

"""
Workflow Registry for the Mathematical Multimodal LLM System.

This module provides a registry for workflow definitions that can be executed
by the Orchestration Manager.
"""
from typing import Dict, Any, List, Optional, Callable, Protocol, Set, Type
import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from ..monitoring.logger import get_logger

logger = get_logger(__name__)


class WorkflowDefinition(ABC):
    """Base class for workflow definitions."""
    
    @abstractmethod
    async def get_initial_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the initial steps for the workflow."""
        pass
        
    @abstractmethod
    async def determine_next_steps(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine the next steps based on completed steps and context."""
        pass
        
    @classmethod
    def get_workflow_type(cls) -> str:
        """Get the workflow type identifier."""
        return cls.__name__.replace("Workflow", "").lower()
        
    @classmethod
    def get_description(cls) -> str:
        """Get the workflow description."""
        return cls.__doc__ or "No description available"


class WorkflowRegistry:
    """Registry for workflow definitions."""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        
    def register_workflow(self, workflow_def: WorkflowDefinition):
        """Register a workflow definition."""
        workflow_type = workflow_def.get_workflow_type()
        self.workflows[workflow_type] = workflow_def
        logger.info(f"Registered workflow: {workflow_type}")
        
    def register_workflow_class(self, workflow_class: Type[WorkflowDefinition]):
        """Register a workflow class by instantiating it."""
        workflow_def = workflow_class()
        self.register_workflow(workflow_def)
        
    def get_workflow(self, workflow_type: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by type."""
        return self.workflows.get(workflow_type)
        
    def has_workflow(self, workflow_type: str) -> bool:
        """Check if a workflow type is registered."""
        return workflow_type in self.workflows
        
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows."""
        return [
            {
                "type": workflow_type,
                "description": workflow_def.get_description()
            }
            for workflow_type, workflow_def in self.workflows.items()
        ]


# Create a singleton instance
_workflow_registry_instance = None

def get_workflow_registry() -> WorkflowRegistry:
    """Get or create the workflow registry singleton instance."""
    global _workflow_registry_instance
    if _workflow_registry_instance is None:
        _workflow_registry_instance = WorkflowRegistry()
    return _workflow_registry_instance


# Standard workflows implementation

class MathProblemSolvingWorkflow(WorkflowDefinition):
    """
    Workflow for solving mathematical problems with step-by-step explanation.
    
    This workflow handles mathematical queries by:
    1. Classifying the query to determine the mathematical domain
    2. Performing symbolic computation
    3. Generating step-by-step solutions
    4. Creating appropriate visualizations
    5. Formatting the response with explanations
    """
    
    async def get_initial_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the initial steps for the math problem solving workflow."""
        # Extract query from context
        query = context.get("query")
        if not query:
            raise ValueError("No query provided in context")
            
        # Define the steps
        steps = [
            {
                "type": "query",
                "name": "classify_query",
                "description": "Classify the mathematical query",
                "agent": "core_llm_agent",
                "capability": "classify_query",
                "parameters": {
                    "query": query,
                    "classify_domain": True,
                    "extract_expressions": True
                },
                "context_keys": []
            }
        ]
        
        return steps
        
    async def determine_next_steps(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine the next steps based on completed steps and context."""
        # Check the last completed step
        if not completed_steps:
            return []
            
        last_step = completed_steps[-1]
        step_name = last_step.get("name")
        
        if step_name == "classify_query":
            # After query classification, perform computation
            domain = context.get("domain", "general")
            expressions = context.get("expressions", [])
            operation = context.get("operation", "solve")
            
            if not expressions:
                return [
                    {
                        "type": "query",
                        "name": "generate_response",
                        "description": "Generate response without computation",
                        "agent": "core_llm_agent",
                        "capability": "generate_response",
                        "parameters": {
                            "query": context.get("query", ""),
                            "response_type": "explanation",
                            "format": "text"
                        },
                        "context_keys": ["domain", "query"]
                    }
                ]
                
            # Define computation step
            next_steps = [
                {
                    "type": "computation",
                    "name": "perform_computation",
                    "description": f"Perform {operation} in domain {domain}",
                    "agent": "math_computation_agent",
                    "capability": "compute",
                    "parameters": {
                        "expression": expressions[0] if expressions else "",
                        "operation": operation,
                        "domain": domain,
                        "step_by_step": True,
                        "format": "latex"
                    },
                    "context_keys": []
                }
            ]
            
            return next_steps
            
        elif step_name == "perform_computation":
            # After computation, generate visualization if appropriate
            domain = context.get("domain", "general")
            result = context.get("result", {})
            
            # Check if visualization would be helpful
            should_visualize = domain in ["calculus", "algebra", "statistics", "geometry"] and "expression" in result
            
            if should_visualize:
                return [
                    {
                        "type": "visualization",
                        "name": "generate_visualization",
                        "description": "Generate visualization for the result",
                        "agent": "visualization_agent",
                        "capability": "generate_visualization",
                        "parameters": {
                            "visualization_type": "function_plot_2d" if domain in ["calculus", "algebra"] else "statistical",
                            "expression": result.get("expression", ""),
                            "domain": domain,
                            "format": "png"
                        },
                        "context_keys": ["result"]
                    }
                ]
            else:
                # Skip visualization and go to response generation
                return [
                    {
                        "type": "query",
                        "name": "generate_response",
                        "description": "Generate final response",
                        "agent": "core_llm_agent",
                        "capability": "generate_response",
                        "parameters": {
                            "include_steps": True,
                            "response_type": "explanation",
                            "format": "latex"
                        },
                        "context_keys": ["query", "domain", "result"]
                    }
                ]
                
        elif step_name == "generate_visualization":
            # After visualization, generate the final response
            return [
                {
                    "type": "query",
                    "name": "generate_response",
                    "description": "Generate final response with visualization",
                    "agent": "core_llm_agent",
                    "capability": "generate_response",
                    "parameters": {
                        "include_steps": True,
                        "include_visualization": True,
                        "response_type": "explanation",
                        "format": "latex"
                    },
                    "context_keys": ["query", "domain", "result", "visualization"]
                }
            ]
            
        elif step_name == "generate_response":
            # This is the final step
            return []
            
        # If we don't recognize the step, return empty list to end workflow
        return []


class HandwritingRecognitionWorkflow(WorkflowDefinition):
    """
    Workflow for processing handwritten mathematical input.
    
    This workflow handles handwritten input by:
    1. Processing the uploaded image
    2. Recognizing mathematical symbols and structure
    3. Converting to LaTeX representation
    4. Performing computation on the recognized expression
    5. Generating appropriate visualization and explanation
    """
    
    async def get_initial_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the initial steps for the handwriting recognition workflow."""
        # Extract image path from context
        image_path = context.get("image_path")
        if not image_path:
            raise ValueError("No image path provided in context")
            
        # Define the steps
        steps = [
            {
                "type": "ocr",
                "name": "recognize_handwriting",
                "description": "Process handwritten mathematical notation",
                "agent": "ocr_agent",
                "capability": "recognize_math",
                "parameters": {
                    "image_path": image_path,
                    "detect_diagrams": True,
                    "enhance_quality": True
                },
                "context_keys": []
            }
        ]
        
        return steps
        
    async def determine_next_steps(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine the next steps based on completed steps and context."""
        # Check the last completed step
        if not completed_steps:
            return []
            
        last_step = completed_steps[-1]
        step_name = last_step.get("name")
        
        if step_name == "recognize_handwriting":
            # After handwriting recognition, classify the expression
            recognized_latex = context.get("recognized_latex", "")
            
            if not recognized_latex:
                return [
                    {
                        "type": "query",
                        "name": "handle_recognition_failure",
                        "description": "Handle recognition failure",
                        "agent": "core_llm_agent",
                        "capability": "generate_response",
                        "parameters": {
                            "response_type": "error",
                            "error_type": "recognition_failure",
                            "format": "text"
                        },
                        "context_keys": []
                    }
                ]
                
            return [
                {
                    "type": "query",
                    "name": "classify_expression",
                    "description": "Classify the recognized expression",
                    "agent": "core_llm_agent",
                    "capability": "classify_query",
                    "parameters": {
                        "latex_expression": recognized_latex,
                        "classify_domain": True,
                        "detect_operation": True
                    },
                    "context_keys": ["recognized_latex"]
                }
            ]
            
        elif step_name == "classify_expression":
            # After classification, perform computation
            recognized_latex = context.get("recognized_latex", "")
            domain = context.get("domain", "general")
            operation = context.get("operation", "evaluate")
            
            return [
                {
                    "type": "computation",
                    "name": "compute_expression",
                    "description": f"Compute the expression in domain {domain}",
                    "agent": "math_computation_agent",
                    "capability": "compute",
                    "parameters": {
                        "expression": recognized_latex,
                        "operation": operation,
                        "domain": domain,
                        "step_by_step": True,
                        "format": "latex"
                    },
                    "context_keys": []
                }
            ]
            
        elif step_name == "compute_expression":
            # After computation, generate visualization
            domain = context.get("domain", "general")
            result = context.get("result", {})
            
            # Check if visualization would be helpful
            should_visualize = domain in ["calculus", "algebra", "statistics", "geometry"] and "expression" in result
            
            if should_visualize:
                return [
                    {
                        "type": "visualization",
                        "name": "generate_visualization",
                        "description": "Generate visualization for the result",
                        "agent": "visualization_agent",
                        "capability": "generate_visualization",
                        "parameters": {
                            "visualization_type": "function_plot_2d" if domain in ["calculus", "algebra"] else "statistical",
                            "expression": result.get("expression", ""),
                            "domain": domain,
                            "format": "png"
                        },
                        "context_keys": ["result"]
                    }
                ]
            else:
                # Skip visualization and go to response generation
                return [
                    {
                        "type": "query",
                        "name": "generate_response",
                        "description": "Generate final response",
                        "agent": "core_llm_agent",
                        "capability": "generate_response",
                        "parameters": {
                            "include_recognized_latex": True,
                            "include_steps": True,
                            "response_type": "explanation",
                            "format": "latex"
                        },
                        "context_keys": ["recognized_latex", "domain", "result"]
                    }
                ]
                
        elif step_name == "generate_visualization":
            # After visualization, generate the final response
            return [
                {
                    "type": "query",
                    "name": "generate_response",
                    "description": "Generate final response with visualization",
                    "agent": "core_llm_agent",
                    "capability": "generate_response",
                    "parameters": {
                        "include_recognized_latex": True,
                        "include_steps": True,
                        "include_visualization": True,
                        "response_type": "explanation",
                        "format": "latex"
                    },
                    "context_keys": ["recognized_latex", "domain", "result", "visualization"]
                }
            ]
            
        elif step_name == "generate_response" or step_name == "handle_recognition_failure":
            # This is the final step
            return []
            
        # If we don't recognize the step, return empty list to end workflow
        return []


# Register standard workflows
def register_standard_workflows():
    """Register standard workflows with the registry."""
    registry = get_workflow_registry()
    
    # Register math problem solving workflow
    registry.register_workflow_class(MathProblemSolvingWorkflow)
    
    # Register handwriting recognition workflow
    registry.register_workflow_class(HandwritingRecognitionWorkflow)


# Auto-register standard workflows when module is imported
register_standard_workflows()

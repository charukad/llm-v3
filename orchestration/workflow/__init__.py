"""
Workflow package for the Mathematical Multimodal LLM System.

This package contains workflow definitions that can be executed
by the Orchestration Manager to process different types of requests.
"""
from .workflow_registry import WorkflowRegistry, WorkflowDefinition, get_workflow_registry
# Import workflows to register them
from . import input_processing_workflow

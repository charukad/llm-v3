"""
Enhanced Workflow Patterns for the Mathematical Multimodal LLM System.

This module provides advanced workflow definitions that support complex
mathematical operations with conditional branching, parallel execution,
and dynamic workflow generation.
"""
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import json
import datetime
import copy

from .workflow_definition import WorkflowDefinition
from ..monitoring.logger import get_logger
from ..monitoring.tracing import get_tracer

logger = get_logger(__name__)


class AdvancedMathWorkflow(WorkflowDefinition):
    """
    Advanced workflow for complex mathematical operations.
    
    This workflow provides enhanced capabilities for handling multi-step
    mathematical operations with conditional branching based on intermediate
    results and domain-specific processing.
    """
    
    @classmethod
    def get_workflow_type(cls) -> str:
        return "advanced_math"
    
    @classmethod
    def get_description(cls) -> str:
        return "Advanced workflow for complex mathematical operations with conditional branching and parallel processing"
    
    async def get_initial_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get the initial steps for the advanced math workflow.
        
        Args:
            context: Workflow context with initial data
            
        Returns:
            List of initial workflow steps
        """
        # Extract query from context
        query = context.get("query")
        if not query:
            raise ValueError("No query provided in context")
            
        # Start with query classification to determine path
        steps = [
            {
                "type": "query",
                "name": "classify_advanced_query",
                "description": "Classify the mathematical query for advanced processing",
                "agent": "core_llm_agent",
                "capability": "classify_query",
                "parameters": {
                    "query": query,
                    "classify_domain": True,
                    "extract_expressions": True,
                    "detect_complexity": True,
                    "identify_sub_problems": True
                },
                "context_keys": []
            }
        ]
        
        return steps
        
    async def determine_next_steps(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determine the next steps based on completed steps and context.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Check if there are any completed steps
        if not completed_steps:
            return []
            
        # Get the last completed step
        last_step = completed_steps[-1]
        step_name = last_step.get("name")
        
        # Handle based on the last completed step
        if step_name == "classify_advanced_query":
            # Query has been classified, determine next steps based on classification
            return await self._handle_query_classification(context, completed_steps)
            
        elif step_name.startswith("decompose_"):
            # Problem decomposition is complete, process sub-problems
            return await self._handle_problem_decomposition(context, completed_steps)
            
        elif step_name.startswith("compute_sub_problem_"):
            # Sub-problem computation is complete, check if all are done
            return await self._handle_sub_problem_completion(context, completed_steps)
            
        elif step_name == "merge_results":
            # Results have been merged, visualize if appropriate
            return await self._handle_results_merging(context, completed_steps)
            
        elif step_name == "generate_advanced_visualization":
            # Visualization is complete, generate response
            return self._create_response_step()
            
        elif step_name == "generate_advanced_response":
            # Final response step complete, end workflow
            return []
            
        # Default behavior for unhandled step types
        logger.warning(f"Unhandled step type in advanced math workflow: {step_name}")
        return []
        
    async def _handle_query_classification(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after query classification.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Extract relevant information from context
        domain = context.get("domain", "general")
        expressions = context.get("expressions", [])
        complexity = context.get("complexity", "simple")
        sub_problems = context.get("sub_problems", [])
        
        # If no expressions were found, we can't proceed with computation
        if not expressions:
            return self._create_response_step()
            
        # For complex queries that can be decomposed into sub-problems
        if complexity in ["complex", "very_complex"] and sub_problems:
            # Create a step to decompose the problem
            return [
                {
                    "type": "query",
                    "name": f"decompose_{domain}_problem",
                    "description": f"Decompose complex {domain} problem into sub-problems",
                    "agent": "core_llm_agent",
                    "capability": "decompose_problem",
                    "parameters": {
                        "domain": domain,
                        "expressions": expressions,
                        "sub_problems": sub_problems,
                        "format": "structured"
                    },
                    "context_keys": ["query", "domain", "complexity"]
                }
            ]
            
        # For single-step computations
        operation = context.get("operation", "evaluate")
        
        # Create a computation step appropriate for the domain and operation
        computation_step = {
            "type": "computation",
            "name": f"compute_{domain}_{operation}",
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
        
        # For domains with specialized handling
        if domain == "linear_algebra":
            # Add matrix handling parameters
            computation_step["parameters"]["matrix_format"] = "latex"
            computation_step["parameters"]["validate_dimensions"] = True
            
        elif domain == "calculus":
            # Add calculus-specific parameters
            computation_step["parameters"]["validation_method"] = "numerical"
            if operation == "integrate":
                computation_step["parameters"]["integration_method"] = "symbolic"
                
        elif domain == "statistics":
            # Add statistics-specific parameters
            computation_step["parameters"]["distribution_check"] = True
            computation_step["parameters"]["confidence_level"] = 0.95
            
        return [computation_step]
        
    async def _handle_problem_decomposition(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after problem decomposition.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Get the decomposed sub-problems from context
        decomposed_problems = context.get("decomposed_problems", [])
        
        if not decomposed_problems:
            logger.warning("No decomposed problems found after decomposition step")
            return self._create_response_step()
            
        # Create computation steps for each sub-problem in parallel
        computation_steps = []
        
        for i, problem in enumerate(decomposed_problems):
            sub_expression = problem.get("expression", "")
            sub_operation = problem.get("operation", "evaluate")
            sub_domain = problem.get("domain", context.get("domain", "general"))
            
            # Skip if no expression
            if not sub_expression:
                continue
                
            # Create a computation step for this sub-problem
            computation_step = {
                "type": "computation",
                "name": f"compute_sub_problem_{i}",
                "description": f"Compute sub-problem {i+1}: {sub_operation} in {sub_domain}",
                "agent": "math_computation_agent",
                "capability": "compute",
                "parameters": {
                    "expression": sub_expression,
                    "operation": sub_operation,
                    "domain": sub_domain,
                    "step_by_step": True,
                    "format": "latex",
                    "sub_problem_id": i
                },
                "context_keys": []
            }
            
            computation_steps.append(computation_step)
            
        # Store the number of sub-problems in context for tracking completion
        context["total_sub_problems"] = len(computation_steps)
        context["completed_sub_problems"] = 0
        context["sub_problem_results"] = {}
        
        return computation_steps
        
    async def _handle_sub_problem_completion(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the completion of a sub-problem computation.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Extract the sub-problem ID from the last step
        last_step = completed_steps[-1]
        step_name = last_step.get("name", "")
        
        # Update completed sub-problems count
        completed_count = context.get("completed_sub_problems", 0) + 1
        context["completed_sub_problems"] = completed_count
        
        # Extract sub-problem ID from step name
        if step_name.startswith("compute_sub_problem_"):
            sub_problem_id = int(step_name.split("_")[-1])
            
            # Store the result for this sub-problem
            if "result" in context:
                context["sub_problem_results"][sub_problem_id] = context["result"]
                
        # Check if all sub-problems are complete
        total_sub_problems = context.get("total_sub_problems", 0)
        
        if completed_count >= total_sub_problems:
            # All sub-problems are complete, merge results
            return [
                {
                    "type": "query",
                    "name": "merge_results",
                    "description": "Merge results from sub-problems",
                    "agent": "core_llm_agent",
                    "capability": "merge_results",
                    "parameters": {
                        "format": "latex"
                    },
                    "context_keys": ["query", "domain", "sub_problem_results", "decomposed_problems"]
                }
            ]
            
        # Not all sub-problems are complete yet, continue processing
        return []
        
    async def _handle_results_merging(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after results merging.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Check if visualization would be helpful
        domain = context.get("domain", "general")
        merged_result = context.get("merged_result", {})
        
        # Determine if visualization would be helpful
        should_visualize = domain in ["calculus", "algebra", "statistics", "geometry"] and \
                           "expression" in merged_result
        
        if should_visualize:
            # Create visualization step
            return [
                {
                    "type": "visualization",
                    "name": "generate_advanced_visualization",
                    "description": "Generate visualization for the merged result",
                    "agent": "visualization_agent",
                    "capability": "generate_visualization",
                    "parameters": {
                        "visualization_type": self._get_visualization_type(domain),
                        "expression": merged_result.get("expression", ""),
                        "domain": domain,
                        "format": "png",
                        "show_steps": True,
                        "include_annotations": True
                    },
                    "context_keys": ["merged_result"]
                }
            ]
        else:
            # Skip visualization and go to response generation
            return self._create_response_step()
            
    def _create_response_step(self) -> List[Dict[str, Any]]:
        """
        Create the final response generation step.
        
        Returns:
            List containing the response step
        """
        return [
            {
                "type": "query",
                "name": "generate_advanced_response",
                "description": "Generate final comprehensive response",
                "agent": "core_llm_agent",
                "capability": "generate_response",
                "parameters": {
                    "include_steps": True,
                    "include_visualization": True,
                    "include_subproblems": True,
                    "response_type": "explanation",
                    "format": "latex"
                },
                "context_keys": ["query", "domain", "merged_result", "visualization", "decomposed_problems", "sub_problem_results"]
            }
        ]
        
    def _get_visualization_type(self, domain: str) -> str:
        """
        Get the appropriate visualization type for a domain.
        
        Args:
            domain: Mathematical domain
            
        Returns:
            Visualization type string
        """
        domain_visualization_map = {
            "algebra": "function_plot_2d",
            "calculus": "function_plot_2d",
            "geometry": "geometric_plot",
            "statistics": "statistical_plot",
            "linear_algebra": "matrix_visualization",
            "probability": "probability_distribution"
        }
        
        return domain_visualization_map.get(domain, "function_plot_2d")


class ParallelProcessingWorkflow(WorkflowDefinition):
    """
    Workflow for parallel processing of mathematical operations.
    
    This workflow enables concurrent execution of multiple mathematical
    operations, improving performance for complex multi-part problems.
    """
    
    @classmethod
    def get_workflow_type(cls) -> str:
        return "parallel_processing"
    
    @classmethod
    def get_description(cls) -> str:
        return "Workflow for parallel processing of multiple mathematical operations"
    
    async def get_initial_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get the initial steps for the parallel processing workflow.
        
        Args:
            context: Workflow context with initial data
            
        Returns:
            List of initial workflow steps
        """
        operations = context.get("operations", [])
        
        if not operations:
            raise ValueError("No operations provided in context")
            
        # Start with query understanding for each operation
        steps = []
        
        for i, operation in enumerate(operations):
            query = operation.get("query", "")
            if not query:
                continue
                
            # Create a classification step for this operation
            step = {
                "type": "query",
                "name": f"classify_operation_{i}",
                "description": f"Classify operation {i+1}",
                "agent": "core_llm_agent",
                "capability": "classify_query",
                "parameters": {
                    "query": query,
                    "classify_domain": True,
                    "extract_expressions": True,
                    "operation_id": i
                },
                "context_keys": []
            }
            
            steps.append(step)
            
        # Store the total number of operations for tracking
        context["total_operations"] = len(steps)
        context["classified_operations"] = {}
        
        return steps
        
    async def determine_next_steps(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determine the next steps based on completed steps and context.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Check if there are any completed steps
        if not completed_steps:
            return []
            
        # Get the last completed step
        last_step = completed_steps[-1]
        step_name = last_step.get("name", "")
        
        # Handle classification steps
        if step_name.startswith("classify_operation_"):
            return await self._handle_operation_classification(context, completed_steps)
            
        # Handle computation steps
        elif step_name.startswith("compute_operation_"):
            return await self._handle_operation_computation(context, completed_steps)
            
        # Handle visualization steps
        elif step_name.startswith("visualize_operation_"):
            return await self._handle_operation_visualization(context, completed_steps)
            
        # Handle merge step
        elif step_name == "merge_parallel_results":
            return self._create_final_response_step()
            
        # Handle final response step
        elif step_name == "generate_parallel_response":
            return []
            
        # Default behavior for unhandled step types
        logger.warning(f"Unhandled step type in parallel processing workflow: {step_name}")
        return []
        
    async def _handle_operation_classification(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after an operation classification.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Extract the operation ID from the last step
        last_step = completed_steps[-1]
        step_name = last_step.get("name", "")
        
        if not step_name.startswith("classify_operation_"):
            return []
            
        operation_id = int(step_name.split("_")[-1])
        
        # Store the classification results for this operation
        classified_operations = context.get("classified_operations", {})
        
        # Extract relevant data from context
        domain = context.get("domain", "general")
        expressions = context.get("expressions", [])
        operation = context.get("operation", "evaluate")
        
        # Store the classification
        classified_operations[operation_id] = {
            "domain": domain,
            "expressions": expressions,
            "operation": operation
        }
        
        context["classified_operations"] = classified_operations
        
        # Check if all operations have been classified
        if len(classified_operations) >= context.get("total_operations", 0):
            # All operations are classified, generate computation steps
            return await self._generate_computation_steps(context)
            
        # Not all operations are classified yet
        return []
        
    async def _generate_computation_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate computation steps for all classified operations.
        
        Args:
            context: Current workflow context
            
        Returns:
            List of computation steps
        """
        classified_operations = context.get("classified_operations", {})
        
        # Create computation steps for all operations
        computation_steps = []
        
        for operation_id, operation_data in classified_operations.items():
            domain = operation_data.get("domain", "general")
            expressions = operation_data.get("expressions", [])
            operation_type = operation_data.get("operation", "evaluate")
            
            if not expressions:
                continue
                
            # Create a computation step for this operation
            step = {
                "type": "computation",
                "name": f"compute_operation_{operation_id}",
                "description": f"Compute operation {operation_id+1}: {operation_type} in {domain}",
                "agent": "math_computation_agent",
                "capability": "compute",
                "parameters": {
                    "expression": expressions[0] if expressions else "",
                    "operation": operation_type,
                    "domain": domain,
                    "step_by_step": True,
                    "format": "latex",
                    "operation_id": operation_id
                },
                "context_keys": []
            }
            
            computation_steps.append(step)
            
        # Initialize tracking for completed computations
        context["total_computations"] = len(computation_steps)
        context["completed_computations"] = 0
        context["computation_results"] = {}
        
        return computation_steps
        
    async def _handle_operation_computation(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the completion of an operation computation.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Extract the operation ID from the last step
        last_step = completed_steps[-1]
        step_name = last_step.get("name", "")
        
        if not step_name.startswith("compute_operation_"):
            return []
            
        operation_id = int(step_name.split("_")[-1])
        
        # Update completed computations count
        completed_count = context.get("completed_computations", 0) + 1
        context["completed_computations"] = completed_count
        
        # Store the result for this operation
        if "result" in context:
            computation_results = context.get("computation_results", {})
            computation_results[operation_id] = context["result"]
            context["computation_results"] = computation_results
            
            # Determine if visualization would be helpful
            domain = context.get("classified_operations", {}).get(operation_id, {}).get("domain", "general")
            should_visualize = domain in ["calculus", "algebra", "statistics", "geometry"] and \
                              "expression" in context["result"]
                              
            if should_visualize:
                # Create visualization step for this operation
                return [
                    {
                        "type": "visualization",
                        "name": f"visualize_operation_{operation_id}",
                        "description": f"Visualize result of operation {operation_id+1}",
                        "agent": "visualization_agent",
                        "capability": "generate_visualization",
                        "parameters": {
                            "visualization_type": self._get_visualization_type(domain),
                            "expression": context["result"].get("expression", ""),
                            "domain": domain,
                            "format": "png",
                            "operation_id": operation_id
                        },
                        "context_keys": []
                    }
                ]
                
        # Check if all computations are complete
        if completed_count >= context.get("total_computations", 0):
            # All computations are complete, merge results
            return [
                {
                    "type": "query",
                    "name": "merge_parallel_results",
                    "description": "Merge results from parallel operations",
                    "agent": "core_llm_agent",
                    "capability": "merge_results",
                    "parameters": {
                        "format": "latex"
                    },
                    "context_keys": ["operations", "classified_operations", "computation_results", "visualization_results"]
                }
            ]
            
        # Not all computations are complete yet
        return []
        
    async def _handle_operation_visualization(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the completion of an operation visualization.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Extract the operation ID from the last step
        last_step = completed_steps[-1]
        step_name = last_step.get("name", "")
        
        if not step_name.startswith("visualize_operation_"):
            return []
            
        operation_id = int(step_name.split("_")[-1])
        
        # Store the visualization for this operation
        if "visualization" in context:
            visualization_results = context.get("visualization_results", {})
            visualization_results[operation_id] = context["visualization"]
            context["visualization_results"] = visualization_results
            
        # Check if all computations are complete to determine if we should merge results
        if context.get("completed_computations", 0) >= context.get("total_computations", 0):
            # All computations are complete, merge results
            return [
                {
                    "type": "query",
                    "name": "merge_parallel_results",
                    "description": "Merge results from parallel operations",
                    "agent": "core_llm_agent",
                    "capability": "merge_results",
                    "parameters": {
                        "format": "latex"
                    },
                    "context_keys": ["operations", "classified_operations", "computation_results", "visualization_results"]
                }
            ]
            
        # Continue processing other operations
        return []
        
    def _create_final_response_step(self) -> List[Dict[str, Any]]:
        """
        Create the final response generation step.
        
        Returns:
            List containing the response step
        """
        return [
            {
                "type": "query",
                "name": "generate_parallel_response",
                "description": "Generate final comprehensive response for all operations",
                "agent": "core_llm_agent",
                "capability": "generate_response",
                "parameters": {
                    "include_steps": True,
                    "include_visualization": True,
                    "response_type": "explanation",
                    "format": "latex"
                },
                "context_keys": ["operations", "merged_results", "computation_results", "visualization_results"]
            }
        ]
        
    def _get_visualization_type(self, domain: str) -> str:
        """
        Get the appropriate visualization type for a domain.
        
        Args:
            domain: Mathematical domain
            
        Returns:
            Visualization type string
        """
        domain_visualization_map = {
            "algebra": "function_plot_2d",
            "calculus": "function_plot_2d",
            "geometry": "geometric_plot",
            "statistics": "statistical_plot",
            "linear_algebra": "matrix_visualization",
            "probability": "probability_distribution"
        }
        
        return domain_visualization_map.get(domain, "function_plot_2d")


class RecoveryWorkflow(WorkflowDefinition):
    """
    Workflow with enhanced error recovery capabilities.
    
    This workflow includes advanced error recovery mechanisms to handle
    failures and provide graceful degradation of functionality.
    """
    
    @classmethod
    def get_workflow_type(cls) -> str:
        return "recovery_enabled"
    
    @classmethod
    def get_description(cls) -> str:
        return "Workflow with enhanced error recovery mechanisms"
    
    async def get_initial_steps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get the initial steps for the recovery-enabled workflow.
        
        Args:
            context: Workflow context with initial data
            
        Returns:
            List of initial workflow steps
        """
        # Extract query from context
        query = context.get("query")
        if not query:
            raise ValueError("No query provided in context")
            
        # Initialize recovery context
        context["recovery"] = {
            "attempts": {},
            "fallbacks": {},
            "errors": []
        }
        
        # Standard initial step for query classification
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
                    "extract_expressions": True,
                    "identify_fallbacks": True  # Request fallback options
                },
                "context_keys": [],
                "recovery_options": {
                    "max_retries": 2,
                    "fallback": "simple_classification"
                }
            }
        ]
        
        return steps
        
    async def determine_next_steps(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determine the next steps based on completed steps and context.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Check if there are any completed steps
        if not completed_steps:
            return []
            
        # Get the last completed step
        last_step = completed_steps[-1]
        step_name = last_step.get("name")
        
        # If the last step has an error, handle recovery
        if "error" in last_step:
            return await self._handle_step_error(context, completed_steps, last_step)
            
        # Regular workflow progression
        if step_name == "classify_query":
            return await self._handle_query_classification(context, completed_steps)
            
        elif step_name == "simple_classification":
            # This is a fallback for classification
            return await self._handle_simple_classification(context, completed_steps)
            
        elif step_name == "perform_computation":
            return await self._handle_computation_completed(context, completed_steps)
            
        elif step_name == "approximate_computation":
            # This is a fallback for computation
            return await self._handle_approximate_computation(context, completed_steps)
            
        elif step_name == "generate_visualization":
            return await self._handle_visualization_completed(context, completed_steps)
            
        elif step_name == "generate_response" or step_name == "generate_fallback_response":
            # Final response step complete, end workflow
            return []
            
        # Default behavior for unhandled step types
        logger.warning(f"Unhandled step type in recovery workflow: {step_name}")
        return []
        
    async def _handle_step_error(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]], failed_step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle errors in workflow steps.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            failed_step: The step that failed
            
        Returns:
            List of next workflow steps for recovery
        """
        step_name = failed_step.get("name")
        error = failed_step.get("error", {})
        recovery_options = failed_step.get("recovery_options", {})
        
        # Record the error
        recovery_context = context.get("recovery", {})
        errors = recovery_context.get("errors", [])
        errors.append({
            "step": step_name,
            "error": error,
            "timestamp": datetime.datetime.now().isoformat()
        })
        recovery_context["errors"] = errors
        context["recovery"] = recovery_context
        
        # Track retry attempts
        attempts = recovery_context.get("attempts", {})
        current_attempts = attempts.get(step_name, 0) + 1
        attempts[step_name] = current_attempts
        recovery_context["attempts"] = attempts
        
        # Check if we should retry
        max_retries = recovery_options.get("max_retries", 1)
        
        if current_attempts <= max_retries:
            # Retry the step
            logger.info(f"Retrying step {step_name} (attempt {current_attempts}/{max_retries})")
            
            # Create a copy of the failed step for retry
            retry_step = copy.deepcopy(failed_step)
            retry_step.pop("error", None)  # Remove the error
            
            # Add retry information to parameters
            retry_step["parameters"]["retry_attempt"] = current_attempts
            
            return [retry_step]
            
        # Max retries exceeded, try fallback
        fallback = recovery_options.get("fallback")
        
        if fallback:
            logger.info(f"Using fallback {fallback} for failed step {step_name}")
            
            # Record the fallback used
            fallbacks = recovery_context.get("fallbacks", {})
            fallbacks[step_name] = fallback
            recovery_context["fallbacks"] = fallbacks
            
            if fallback == "simple_classification":
                return self._create_simple_classification_step(context)
                
            elif fallback == "approximate_computation":
                return self._create_approximate_computation_step(context)
                
            elif fallback == "skip_visualization":
                return self._create_response_step(context, True)
                
            elif fallback == "minimal_response":
                return self._create_fallback_response_step(context)
                
        # No fallback available or all recovery options exhausted
        logger.error(f"All recovery options exhausted for step {step_name}")
        
        # Try to proceed with a fallback response
        return self._create_fallback_response_step(context)
        
    async def _handle_query_classification(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after query classification.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Extract relevant information from context
        domain = context.get("domain", "general")
        expressions = context.get("expressions", [])
        operation = context.get("operation", "evaluate")
        
        # If no expressions were found, go directly to response
        if not expressions:
            return self._create_response_step(context, False)
            
        # Create computation step
        computation_step = {
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
            "context_keys": [],
            "recovery_options": {
                "max_retries": 2,
                "fallback": "approximate_computation"
            }
        }
        
        return [computation_step]
        
    async def _handle_simple_classification(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after simple classification fallback.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Extract relevant information from context
        domain = context.get("simple_domain", context.get("domain", "general"))
        expressions = context.get("simple_expressions", context.get("expressions", []))
        operation = context.get("simple_operation", context.get("operation", "evaluate"))
        
        # Update context with simplified information
        context["domain"] = domain
        context["expressions"] = expressions
        context["operation"] = operation
        
        # If no expressions were found, go directly to response
        if not expressions:
            return self._create_response_step(context, True)
            
        # Create computation step (with simpler expectations)
        computation_step = {
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
                "format": "latex",
                "simplified": True
            },
            "context_keys": [],
            "recovery_options": {
                "max_retries": 1,
                "fallback": "approximate_computation"
            }
        }
        
        return [computation_step]
        
    async def _handle_computation_completed(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after computation is completed.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Check if visualization would be helpful
        domain = context.get("domain", "general")
        result = context.get("result", {})
        
        # Determine if visualization would be helpful
        should_visualize = domain in ["calculus", "algebra", "statistics", "geometry"] and \
                         "expression" in result
        
        if should_visualize:
            # Create visualization step
            return [
                {
                    "type": "visualization",
                    "name": "generate_visualization",
                    "description": "Generate visualization for the result",
                    "agent": "visualization_agent",
                    "capability": "generate_visualization",
                    "parameters": {
                        "visualization_type": self._get_visualization_type(domain),
                        "expression": result.get("expression", ""),
                        "domain": domain,
                        "format": "png"
                    },
                    "context_keys": ["result"],
                    "recovery_options": {
                        "max_retries": 1,
                        "fallback": "skip_visualization"
                    }
                }
            ]
        else:
            # Skip visualization and go to response generation
            return self._create_response_step(context, False)
            
    async def _handle_approximate_computation(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after approximate computation fallback.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Skip visualization for approximate results and go to response
        return self._create_response_step(context, True)
        
    async def _handle_visualization_completed(self, context: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle the next steps after visualization is completed.
        
        Args:
            context: Current workflow context
            completed_steps: List of completed workflow steps
            
        Returns:
            List of next workflow steps
        """
        # Generate the final response
        return self._create_response_step(context, False)
        
    def _create_simple_classification_step(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a simplified classification step for fallback.
        
        Args:
            context: Current workflow context
            
        Returns:
            List containing the simple classification step
        """
        query = context.get("query", "")
        
        return [
            {
                "type": "query",
                "name": "simple_classification",
                "description": "Simple classification of the mathematical query",
                "agent": "core_llm_agent",
                "capability": "classify_query",
                "parameters": {
                    "query": query,
                    "classify_domain": True,
                    "extract_expressions": True,
                    "simplified": True
                },
                "context_keys": [],
                "recovery_options": {
                    "max_retries": 1,
                    "fallback": "minimal_response"
                }
            }
        ]
        
    def _create_approximate_computation_step(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create an approximate computation step for fallback.
        
        Args:
            context: Current workflow context
            
        Returns:
            List containing the approximate computation step
        """
        domain = context.get("domain", "general")
        expressions = context.get("expressions", [])
        operation = context.get("operation", "evaluate")
        
        if not expressions:
            return self._create_fallback_response_step(context)
            
        return [
            {
                "type": "computation",
                "name": "approximate_computation",
                "description": f"Perform approximate {operation} in domain {domain}",
                "agent": "math_computation_agent",
                "capability": "compute",
                "parameters": {
                    "expression": expressions[0] if expressions else "",
                    "operation": operation,
                    "domain": domain,
                    "step_by_step": False,
                    "format": "latex",
                    "approximate": True,
                    "numerical_fallback": True
                },
                "context_keys": [],
                "recovery_options": {
                    "max_retries": 0,
                    "fallback": "minimal_response"
                }
            }
        ]
        
    def _create_response_step(self, context: Dict[str, Any], is_fallback: bool) -> List[Dict[str, Any]]:
        """
        Create a response generation step.
        
        Args:
            context: Current workflow context
            is_fallback: Whether this is a fallback response
            
        Returns:
            List containing the response step
        """
        return [
            {
                "type": "query",
                "name": "generate_response",
                "description": "Generate final response",
                "agent": "core_llm_agent",
                "capability": "generate_response",
                "parameters": {
                    "include_steps": True,
                    "include_visualization": "visualization" in context,
                    "response_type": "explanation",
                    "format": "latex",
                    "is_fallback": is_fallback
                },
                "context_keys": ["query", "domain", "result", "visualization", "recovery"],
                "recovery_options": {
                    "max_retries": 1,
                    "fallback": "minimal_response"
                }
            }
        ]
        
    def _create_fallback_response_step(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a minimal fallback response step.
        
        Args:
            context: Current workflow context
            
        Returns:
            List containing the fallback response step
        """
        return [
            {
                "type": "query",
                "name": "generate_fallback_response",
                "description": "Generate minimal fallback response",
                "agent": "core_llm_agent",
                "capability": "generate_response",
                "parameters": {
                    "include_steps": False,
                    "include_visualization": False,
                    "response_type": "fallback",
                    "format": "text",
                    "is_fallback": True,
                    "include_error_details": True
                },
                "context_keys": ["query", "recovery"]
            }
        ]
        
    def _get_visualization_type(self, domain: str) -> str:
        """
        Get the appropriate visualization type for a domain.
        
        Args:
            domain: Mathematical domain
            
        Returns:
            Visualization type string
        """
        domain_visualization_map = {
            "algebra": "function_plot_2d",
            "calculus": "function_plot_2d",
            "geometry": "geometric_plot",
            "statistics": "statistical_plot",
            "linear_algebra": "matrix_visualization",
            "probability": "probability_distribution"
        }
        
        return domain_visualization_map.get(domain, "function_plot_2d")


# Register enhanced workflows
def register_enhanced_workflows():
    """Register enhanced workflows with the registry."""
    from .workflow_registry import get_workflow_registry
    
    registry = get_workflow_registry()
    
    # Register advanced math workflow
    registry.register_workflow_class(AdvancedMathWorkflow)
    
    # Register parallel processing workflow
    registry.register_workflow_class(ParallelProcessingWorkflow)
    
    # Register recovery workflow
    registry.register_workflow_class(RecoveryWorkflow)
    
    logger.info("Enhanced workflows registered")


# Auto-register enhanced workflows when module is imported
register_enhanced_workflows()

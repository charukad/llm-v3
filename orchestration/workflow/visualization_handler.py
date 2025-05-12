"""
Workflow handler for visualization workflows.

This module provides specialized handling for visualization workflows,
integrating with the Orchestration Manager.
"""

from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

class VisualizationWorkflowHandler:
    """
    Handler for visualization workflows in the Orchestration Manager.
    """
    
    def __init__(self, orchestration_manager):
        """
        Initialize the visualization workflow handler.
        
        Args:
            orchestration_manager: The Orchestration Manager instance
        """
        self.orchestration_manager = orchestration_manager
        self.agent_registry = orchestration_manager.agent_registry
    
    def start_visualization_workflow(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a visualization generation workflow.
        
        Args:
            request: Visualization request data
            
        Returns:
            Dictionary with workflow information
        """
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Extract request data
        mathematical_context = request.get("mathematical_context", {})
        visualization_type = request.get("visualization_type")
        parameters = request.get("parameters", {})
        interaction_id = request.get("interaction_id")
        
        # Initialize workflow data
        workflow_data = {
            "mathematical_context": mathematical_context,
            "visualization_type": visualization_type,
            "parameters": parameters,
            "interaction_id": interaction_id,
            "workflow_id": workflow_id,
            "start_time": datetime.now().isoformat()
        }
        
        # Start the workflow
        self.orchestration_manager.start_workflow(
            workflow_type="visualization_generation",
            workflow_id=workflow_id,
            initial_data=workflow_data
        )
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "Visualization workflow started"
        }
    
    def start_math_visualization_workflow(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a workflow to enhance a mathematical response with visualizations.
        
        Args:
            request: Request data containing mathematical response
            
        Returns:
            Dictionary with workflow information
        """
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Extract request data
        mathematical_response = request.get("mathematical_response", {})
        interaction_id = request.get("interaction_id")
        max_visualizations = request.get("max_visualizations", 3)
        
        # Initialize workflow data
        workflow_data = {
            "mathematical_response": mathematical_response,
            "interaction_id": interaction_id,
            "max_visualizations": max_visualizations,
            "workflow_id": workflow_id,
            "start_time": datetime.now().isoformat(),
            "visualizations": []
        }
        
        # Start the workflow
        self.orchestration_manager.start_workflow(
            workflow_type="math_response_visualization",
            workflow_id=workflow_id,
            initial_data=workflow_data
        )
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "Mathematical response visualization workflow started"
        }
    
    def handle_visualization_step(self, step_name: str, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a specific step in a visualization workflow.
        
        Args:
            step_name: Name of the workflow step
            workflow_data: Current workflow data
            
        Returns:
            Updated workflow data
        """
        # Handle based on step name
        if step_name == "parse_request":
            return self._handle_parse_request(workflow_data)
        elif step_name == "determine_visualization_type":
            return self._handle_determine_visualization(workflow_data)
        elif step_name == "generate_visualization":
            return self._handle_generate_visualization(workflow_data)
        elif step_name == "format_response":
            return self._handle_format_response(workflow_data)
        elif step_name == "analyze_math_response":
            return self._handle_analyze_math_response(workflow_data)
        elif step_name == "extract_visualization_contexts":
            return self._handle_extract_visualization_contexts(workflow_data)
        elif step_name == "determine_visualizations":
            return self._handle_determine_visualizations(workflow_data)
        elif step_name == "generate_visualizations":
            return self._handle_generate_visualizations(workflow_data)
        elif step_name == "enhance_response":
            return self._handle_enhance_response(workflow_data)
        else:
            # Unknown step, return data unchanged
            return workflow_data
    
    def _handle_parse_request(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the parse_request step."""
        # Basic validation
        if "mathematical_context" not in workflow_data:
            workflow_data["error"] = "Missing mathematical context"
            workflow_data["status"] = "error"
            return workflow_data
        
        # If visualization type is not provided, mark for determination
        if not workflow_data.get("visualization_type"):
            workflow_data["needs_visualization_determination"] = True
        
        workflow_data["status"] = "in_progress"
        return workflow_data
    
    def _handle_determine_visualization(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the determine_visualization_type step."""
        # This step would normally be handled by the visualization agent
        # We're just providing a placeholder for orchestration testing
        
        # If visualization type was already specified, use it
        if workflow_data.get("visualization_type"):
            workflow_data["status"] = "visualization_determined"
            return workflow_data
        
        # Get context for visualization selection
        context = workflow_data.get("mathematical_context", {})
        
        # Get visualization selector agent
        agent = self.agent_registry.get_agent_by_capability("determine_visualization_type")
        if not agent:
            workflow_data["error"] = "Visualization selector agent not found"
            workflow_data["status"] = "error"
            return workflow_data
        
        # Create message for agent
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "orchestration_manager",
                "recipient": agent["id"],
                "timestamp": datetime.now().isoformat(),
                "message_type": "visualization_selection_request"
            },
            "body": context
        }
        
        # Send message to agent and get response
        response = self.orchestration_manager.message_bus.send_request_sync(
            agent["id"], 
            message
        )
        
        # Process response
        if response and response.get("body", {}).get("success", False):
            recommended = response["body"].get("recommended_visualization", {})
            workflow_data["visualization_type"] = recommended.get("type")
            workflow_data["parameters"].update(recommended.get("params", {}))
            workflow_data["alternative_visualizations"] = response["body"].get("alternative_visualizations", [])
            workflow_data["status"] = "visualization_determined"
        else:
            workflow_data["error"] = "Failed to determine visualization type"
            workflow_data["status"] = "error"
        
        return workflow_data
    
    def _handle_generate_visualization(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the generate_visualization step."""
        # This step would normally be handled by the visualization agent
        # We're just providing a placeholder for orchestration testing
        
        # Get visualization type and parameters
        visualization_type = workflow_data.get("visualization_type")
        parameters = workflow_data.get("parameters", {})
        
        if not visualization_type:
            workflow_data["error"] = "No visualization type specified"
            workflow_data["status"] = "error"
            return workflow_data
        
        # Determine which agent to use based on visualization type
        agent_type = "visualization"
        advanced_types = ["derivative", "critical_points", "integral", "taylor_series", "vector_field"]
        
        if visualization_type in advanced_types:
            agent_type = "advanced_visualization"
        
        # Get appropriate agent
        agent = self.agent_registry.get_agent_by_type(agent_type)
        if not agent:
            workflow_data["error"] = f"{agent_type} agent not found"
            workflow_data["status"] = "error"
            return workflow_data
        
        # Create message for agent
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "orchestration_manager",
                "recipient": agent["id"],
                "timestamp": datetime.now().isoformat(),
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": visualization_type,
                "parameters": parameters
            }
        }
        
        # Send message to agent and get response
        response = self.orchestration_manager.message_bus.send_request_sync(
            agent["id"], 
            message
        )
        
        # Process response
        if response and response.get("body", {}).get("success", False):
            workflow_data["visualization_result"] = response["body"]
            workflow_data["status"] = "visualization_generated"
        else:
            workflow_data["error"] = "Failed to generate visualization"
            workflow_data["error_details"] = response.get("body", {}).get("error") if response else "No response"
            workflow_data["status"] = "error"
        
        return workflow_data
    
    def _handle_format_response(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the format_response step."""
        # Format the final response with visualization data
        result = workflow_data.get("visualization_result", {})
        
        # Create formatted response
        response = {
            "success": True,
            "visualization_type": workflow_data.get("visualization_type"),
            "result": result,
            "workflow_id": workflow_data.get("workflow_id")
        }
        
        # If there was an error, include it
        if "error" in workflow_data:
            response["success"] = False
            response["error"] = workflow_data["error"]
            response["error_details"] = workflow_data.get("error_details")
        
        workflow_data["formatted_response"] = response
        workflow_data["status"] = "completed"
        
        return workflow_data
    
    def _handle_analyze_math_response(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the analyze_math_response step."""
        # Basic validation
        if "mathematical_response" not in workflow_data:
            workflow_data["error"] = "Missing mathematical response"
            workflow_data["status"] = "error"
            return workflow_data
        
        # Mark for context extraction
        workflow_data["needs_context_extraction"] = True
        workflow_data["status"] = "in_progress"
        
        return workflow_data
    
    def _handle_extract_visualization_contexts(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the extract_visualization_contexts step."""
        # This would normally be handled by the Core LLM Agent
        # We're providing a placeholder for testing
        
        # In a real implementation, the Core LLM Agent would analyze the
        # mathematical response and extract contexts for visualization
        
        # For testing, create a simple context based on the response
        response = workflow_data.get("mathematical_response", {})
        
        # Extract mathematical expressions (placeholder logic)
        expressions = []
        if "expressions" in response:
            expressions = response["expressions"]
        elif "latex_expressions" in response:
            expressions = response["latex_expressions"]
        
        # Create visualization contexts
        contexts = []
        for i, expr in enumerate(expressions[:3]):  # Limit to first 3 expressions
            contexts.append({
                "context_id": f"ctx_{i}",
                "expression": expr,
                "domain": response.get("domain", "general"),
                "operation": response.get("operation", ""),
                "visualization_priority": i == 0  # Prioritize first expression
            })
        
        workflow_data["visualization_contexts"] = contexts
        workflow_data["status"] = "contexts_extracted"
        
        return workflow_data
    
    def _handle_determine_visualizations(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the determine_visualizations step."""
        # Get contexts for visualization selection
        contexts = workflow_data.get("visualization_contexts", [])
        
        if not contexts:
            workflow_data["status"] = "no_visualizations_needed"
            return workflow_data
        
        # Get visualization selector agent
        agent = self.agent_registry.get_agent_by_capability("determine_visualization_type")
        if not agent:
            workflow_data["error"] = "Visualization selector agent not found"
            workflow_data["status"] = "error"
            return workflow_data
        
        # Process each context
        visualization_tasks = []
        
        for context in contexts:
            # Create message for agent
            message = {
                "header": {
                    "message_id": str(uuid.uuid4()),
                    "sender": "orchestration_manager",
                    "recipient": agent["id"],
                    "timestamp": datetime.now().isoformat(),
                    "message_type": "visualization_selection_request"
                },
                "body": context
            }
            
            # Send message to agent and get response
            response = self.orchestration_manager.message_bus.send_request_sync(
                agent["id"], 
                message
            )
            
            # Process response
            if response and response.get("body", {}).get("success", False):
                recommended = response["body"].get("recommended_visualization", {})
                visualization_tasks.append({
                    "context_id": context.get("context_id"),
                    "visualization_type": recommended.get("type"),
                    "parameters": recommended.get("params", {}),
                    "priority": context.get("visualization_priority", False)
                })
        
        workflow_data["visualization_tasks"] = visualization_tasks
        workflow_data["status"] = "visualizations_determined"
        
        return workflow_data
    
    def _handle_generate_visualizations(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the generate_visualizations step."""
        # Get visualization tasks
        tasks = workflow_data.get("visualization_tasks", [])
        
        if not tasks:
            workflow_data["status"] = "no_visualizations_needed"
            return workflow_data
        
        # Get max visualizations limit
        max_visualizations = workflow_data.get("max_visualizations", 3)
        
        # Sort tasks by priority
        tasks.sort(key=lambda x: not x.get("priority", False))
        
        # Limit to max visualizations
        tasks = tasks[:max_visualizations]
        
        # Generate each visualization
        generated_visualizations = []
        
        for task in tasks:
            visualization_type = task.get("visualization_type")
            parameters = task.get("parameters", {})
            
            if not visualization_type:
                continue
            
            # Determine which agent to use based on visualization type
            agent_type = "visualization"
            advanced_types = ["derivative", "critical_points", "integral", "taylor_series", "vector_field"]
            
            if visualization_type in advanced_types:
                agent_type = "advanced_visualization"
            
            # Get appropriate agent
            agent = self.agent_registry.get_agent_by_type(agent_type)
            if not agent:
                continue
            
            # Add interaction ID if available
            if "interaction_id" in workflow_data:
                parameters["interaction_id"] = workflow_data["interaction_id"]
            
            # Create message for agent
            message = {
                "header": {
                    "message_id": str(uuid.uuid4()),
                    "sender": "orchestration_manager",
                    "recipient": agent["id"],
                    "timestamp": datetime.now().isoformat(),
                    "message_type": "visualization_request"
                },
                "body": {
                    "visualization_type": visualization_type,
                    "parameters": parameters
                }
            }
            
            # Send message to agent and get response
            response = self.orchestration_manager.message_bus.send_request_sync(
                agent["id"], 
                message
            )
            
            # Process response
            if response and response.get("body", {}).get("success", False):
                generated_visualizations.append({
                    "context_id": task.get("context_id"),
                    "visualization_type": visualization_type,
                    "result": response["body"],
                    "parameters": parameters
                })
        
        workflow_data["visualizations"] = generated_visualizations
        workflow_data["status"] = "visualizations_generated"
        
        return workflow_data
    
    def _handle_enhance_response(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the enhance_response step."""
        # This would normally be handled by the Core LLM Agent
        # We're providing a placeholder for testing
        
        # Get original response and generated visualizations
        original_response = workflow_data.get("mathematical_response", {})
        visualizations = workflow_data.get("visualizations", [])
        
        # Create a simple enhanced response
        enhanced_response = dict(original_response)
        
        if visualizations:
            # Add visualization references to response
            enhanced_response["visualizations"] = visualizations
            enhanced_response["enhanced_with_visualizations"] = True
            
            # Create simple text explaining visualizations
            visualization_text = "I've created the following visualizations to help illustrate this:\n\n"
            for i, viz in enumerate(visualizations):
                viz_type = viz.get("visualization_type", "visualization")
                viz_result = viz.get("result", {})
                viz_id = viz_result.get("visualization_id", f"viz_{i}")
                
                visualization_text += f"- {viz_type.capitalize()} (ID: {viz_id})\n"
            
            # Add to response
            if "text" in enhanced_response:
                enhanced_response["text"] += f"\n\n{visualization_text}"
            else:
                enhanced_response["text"] = visualization_text
        
        workflow_data["enhanced_response"] = enhanced_response
        workflow_data["status"] = "completed"
        
        return workflow_data

"""
End-to-end workflow definitions for the Mathematical Multimodal LLM System.

This module defines comprehensive workflows that utilize all system components
to process mathematical queries from start to finish.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import time

from ..agents.registry import AgentRegistry
from ..message_bus.message_formats import create_message
from core.agent.multimodal_integration import MultimodalLLMIntegration
from math_processing.agent.math_multimodal_integration import MathMultimodalIntegration
from multimodal.context.context_manager import ContextManager
from multimodal.unified_pipeline.input_processor import InputProcessor
from multimodal.unified_pipeline.content_router import ContentRouter

logger = logging.getLogger(__name__)

class EndToEndWorkflowManager:
    """Manager for end-to-end workflows in the system."""
    
    def __init__(self, registry: AgentRegistry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the end-to-end workflow manager.
        
        Args:
            registry: Agent registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        
        # Get required services and agents
        self.context_manager = self._get_service("context_manager")
        self.input_processor = self._get_service("input_processor")
        self.content_router = self._get_service("content_router")
        
        self.core_llm_agent = self._get_agent("core_llm_agent")
        self.math_agent = self._get_agent("math_computation_agent")
        
        # Initialize integrations
        self.llm_integration = MultimodalLLMIntegration(self.core_llm_agent)
        self.math_integration = MathMultimodalIntegration(self.math_agent)
        
        # Workflow storage
        self.active_workflows = {}
        
        logger.info("Initialized end-to-end workflow manager")
    
    def _get_service(self, service_id: str) -> Any:
        """
        Get a service instance from the registry.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service instance
        """
        service_info = self.registry.get_service_info(service_id)
        if not service_info or "instance" not in service_info:
            raise ValueError(f"Service not found or not initialized: {service_id}")
        return service_info["instance"]
    
    def _get_agent(self, agent_id: str) -> Any:
        """
        Get an agent instance from the registry.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance
        """
        if agent_id not in self.registry.agents or "instance" not in self.registry.agents[agent_id]:
            raise ValueError(f"Agent not found or not initialized: {agent_id}")
        return self.registry.agents[agent_id]["instance"]
    
    def start_workflow(self, input_data: Dict[str, Any], 
                      context_id: Optional[str] = None,
                      conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a new end-to-end workflow.
        
        Args:
            input_data: Input data for the workflow
            context_id: Optional context ID
            conversation_id: Optional conversation ID
            
        Returns:
            Dictionary containing workflow information
        """
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Create context if needed
        if not context_id:
            context = self.context_manager.create_context(conversation_id)
            context_id = context.context_id
        
        # Create workflow state
        workflow = {
            "id": workflow_id,
            "type": "end_to_end",
            "state": "initializing",
            "input_data": input_data,
            "context_id": context_id,
            "conversation_id": conversation_id,
            "steps": [],
            "current_step": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store in active workflows
        self.active_workflows[workflow_id] = workflow
        
        # Start asynchronous processing
        self._process_workflow_async(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "context_id": context_id,
            "conversation_id": conversation_id,
            "state": "initializing",
            "message": "Workflow started successfully"
        }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dictionary containing workflow status
        """
        if workflow_id not in self.active_workflows:
            return {
                "success": False,
                "error": f"Workflow not found: {workflow_id}"
            }
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "state": workflow["state"],
            "current_step": workflow["current_step"],
            "steps_completed": len(workflow["steps"]),
            "created_at": workflow["created_at"],
            "updated_at": workflow["updated_at"]
        }
    
    def get_workflow_result(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dictionary containing workflow result
        """
        if workflow_id not in self.active_workflows:
            return {
                "success": False,
                "error": f"Workflow not found: {workflow_id}"
            }
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow["state"] != "completed":
            return {
                "success": False,
                "error": f"Workflow not completed: {workflow_id}",
                "state": workflow["state"]
            }
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "result": workflow.get("result", {}),
            "steps": workflow["steps"]
        }
    
    def _process_workflow_async(self, workflow_id: str) -> None:
        """
        Process a workflow asynchronously.
        
        In a production system, this would be a background task.
        
        Args:
            workflow_id: Workflow identifier
        """
        # In a real implementation, this would be handled by a task queue or threads
        # For this example, we'll just process it sequentially
        
        try:
            workflow = self.active_workflows[workflow_id]
            workflow["state"] = "processing"
            workflow["steps"] = []
            
            # Step 1: Process the input
            self._update_workflow_step(workflow_id, "input_processing", "Processing input")
            input_data = workflow["input_data"]
            
            # Determine input type and process accordingly
            if "input_type" in input_data:
                # Already processed
                processed_input = input_data
            elif "content" in input_data:
                # Raw content
                content = input_data["content"]
                content_type = input_data.get("content_type", "text/plain")
                processed_input = self.input_processor.process_input(content, content_type)
            else:
                raise ValueError("Invalid input data format")
            
            # Add processed input to workflow
            workflow["processed_input"] = processed_input
            
            # Step 2: Check for ambiguities
            self._update_workflow_step(workflow_id, "ambiguity_checking", "Checking for ambiguities")
            
            # In a real implementation, this would check for ambiguities
            # For this example, we'll assume no ambiguities
            
            # Step 3: Route content to appropriate agents
            self._update_workflow_step(workflow_id, "content_routing", "Routing content to appropriate agents")
            
            routing_result = self.content_router.route_content(processed_input)
            workflow["routing_result"] = routing_result
            
            # Step 4: Process with appropriate agents
            self._update_workflow_step(workflow_id, "agent_processing", "Processing with specialized agents")
            
            # Handle different agent types
            agent_type = routing_result.get("agent_type", "core_llm")
            agent_result = None
            
            if agent_type == "ocr" or agent_type == "advanced_ocr":
                # Already processed by the input pipeline
                agent_result = routing_result.get("result", {})
                
            elif agent_type == "core_llm":
                # Initial processing with LLM
                agent_result = self.llm_integration.process_multimodal_input(
                    processed_input,
                    {"conversation_id": workflow.get("conversation_id")}
                )
            
            # Add agent result to workflow
            workflow["agent_result"] = agent_result
            
            # Step 5: Mathematical processing if needed
            math_result = None
            contains_math = agent_result.get("contains_math", False) if agent_result else False
            
            if contains_math:
                self._update_workflow_step(workflow_id, "math_processing", "Processing mathematical content")
                
                math_result = self.math_integration.process_multimodal_input(processed_input)
                workflow["math_result"] = math_result
            
            # Step 6: Generate final response
            self._update_workflow_step(workflow_id, "response_generation", "Generating final response")
            
            final_response = None
            
            if math_result and math_result.get("success", False):
                # Generate response with mathematical results
                final_response = self.llm_integration.process_with_mathematical_result(
                    processed_input,
                    math_result,
                    {"conversation_id": workflow.get("conversation_id")}
                )
            else:
                # Use the initial response
                final_response = agent_result
            
            # Step 7: Update context
            self._update_workflow_step(workflow_id, "context_update", "Updating context")
            
            # Add to context
            context_id = workflow["context_id"]
            context = self.context_manager.get_context(context_id)
            
            if context:
                # Add input entity
                input_entity_id = self.context_manager.add_entity_to_context(
                    context_id,
                    {
                        "type": "input",
                        "content": processed_input,
                        "timestamp": datetime.now().isoformat()
                    },
                    processed_input.get("input_type", "text")
                )
                
                # Add response entity
                response_entity_id = self.context_manager.add_entity_to_context(
                    context_id,
                    {
                        "type": "response",
                        "content": final_response,
                        "timestamp": datetime.now().isoformat()
                    },
                    "text"
                )
                
                # Add relation between input and response
                if input_entity_id and response_entity_id:
                    self.context_manager.add_reference_to_context(
                        context_id,
                        input_entity_id,
                        response_entity_id,
                        "response_to"
                    )
            
            # Finalize workflow
            workflow["result"] = final_response
            workflow["state"] = "completed"
            workflow["updated_at"] = datetime.now().isoformat()
            
            logger.info(f"Workflow completed successfully: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Error processing workflow {workflow_id}: {str(e)}")
            
            # Update workflow state
            workflow = self.active_workflows.get(workflow_id)
            if workflow:
                workflow["state"] = "error"
                workflow["error"] = str(e)
                workflow["updated_at"] = datetime.now().isoformat()
    
    def _update_workflow_step(self, workflow_id: str, step_name: str, description: str) -> None:
        """
        Update workflow with a new step.
        
        Args:
            workflow_id: Workflow identifier
            step_name: Step name
            description: Step description
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return
        
        # Add step
        workflow["steps"].append({
            "name": step_name,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update current step
        workflow["current_step"] = len(workflow["steps"])
        workflow["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Workflow {workflow_id} - Step: {step_name}")

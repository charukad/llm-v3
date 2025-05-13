"""
Orchestration Manager for the Mathematical Multimodal LLM System.

This module provides the central orchestration logic for coordinating
workflows between different agents in the system.
"""
import asyncio
import datetime
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union, TypeVar, Tuple
import logging
import json
import time
from pydantic import BaseModel, Field

from ..message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message,
    create_computation_request, create_visualization_request, create_error_response
)
from ..message_bus.rabbitmq_wrapper import get_message_bus
from ..monitoring.logger import get_logger
from ..monitoring.tracing import get_tracer, Span
from ..monitoring.metrics import get_registry, record_processing_time
from ..workflow.workflow_registry import WorkflowRegistry, get_workflow_registry
from ..agents.registry import AgentRegistry, get_agent_registry

logger = get_logger(__name__)


class WorkflowStatus(str, Enum):
    """Status of a workflow."""
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class WorkflowContext(BaseModel):
    """Context for a workflow execution."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[str] = None
    workflow_type: str
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    completed_at: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    current_step_index: int = 0
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_status(self, status: WorkflowStatus):
        """Update the status of the workflow."""
        self.status = status
        self.updated_at = datetime.datetime.now().isoformat()
        if status == WorkflowStatus.COMPLETED or status == WorkflowStatus.FAILED:
            self.completed_at = datetime.datetime.now().isoformat()
            
    def add_step(self, step: Dict[str, Any]):
        """Add a step to the workflow."""
        self.steps.append(step)
        
    def update_data(self, data: Dict[str, Any]):
        """Update the workflow data."""
        self.data.update(data)
        self.updated_at = datetime.datetime.now().isoformat()
        
    def set_error(self, error_message: str, error_code: str = "WORKFLOW_ERROR", details: Dict[str, Any] = None):
        """Set an error on the workflow."""
        self.error = {
            "message": error_message,
            "code": error_code,
            "details": details or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.update_status(WorkflowStatus.FAILED)


class OrchestrationManager:
    """
    Orchestration Manager for coordinating workflows between agents.
    """
    def __init__(self):
        self.message_bus = get_message_bus()
        self.workflow_registry = get_workflow_registry()
        self.agent_registry = get_agent_registry()
        
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_futures: Dict[str, asyncio.Future] = {}
        self.workflow_subscription_callbacks: Dict[str, List[Callable]] = {}
        
        self.metrics_registry = get_registry()
        self.tracer = get_tracer()
        
        # Initialize metrics
        self._setup_metrics()
        
    async def initialize(self):
        """Initialize the orchestration manager."""
        # Connect to the message bus if not already connected
        if not self.message_bus.connection:
            await self.message_bus.connect()
            
        # Set up the orchestration queue
        orchestration_queue = await self.message_bus.declare_queue(
            "orchestration.manager",
            durable=True
        )
        
        # Bind to response and status update messages
        await self.message_bus.bind_queue(
            "orchestration.manager",
            "agent.orchestration_manager"
        )
        
        # Set up consumer
        await self.message_bus.setup_consumer(
            "orchestration.manager",
            self._handle_message
        )
        
        # Register the orchestration manager with the router
        self.message_bus.router.register_agent(
            "orchestration_manager",
            ["workflow_management", "orchestration"]
        )
        
        # Register for broadcast notifications
        self.message_bus.add_message_listener(
            MessageType.STATUS_UPDATE,
            self._handle_status_update
        )
        
        logger.info("Orchestration manager initialized")
        
    def _setup_metrics(self):
        """Set up metrics for the orchestration manager."""
        registry = self.metrics_registry
        
        # Workflow metrics
        registry.counter("orchestration.workflows.total", "Total number of workflows started")
        registry.counter("orchestration.workflows.completed", "Number of completed workflows")
        registry.counter("orchestration.workflows.failed", "Number of failed workflows")
        
        # Workflow gauge
        registry.gauge("orchestration.workflows.active", "Number of active workflows")
        
        # Step metrics
        registry.counter("orchestration.steps.total", "Total number of workflow steps executed")
        registry.counter("orchestration.steps.failed", "Number of failed workflow steps")
        
        # Latency histogram
        registry.histogram(
            "orchestration.workflow.duration",
            "Workflow execution duration (ms)",
            buckets=[100, 500, 1000, 2500, 5000, 10000, 30000, 60000, 300000]
        )
        
    async def _handle_message(self, message: Message):
        """Handle messages sent to the orchestration manager."""
        message_type = message.header.message_type
        
        # Handle different message types
        if message_type in [MessageType.COMPUTATION_RESULT, MessageType.VISUALIZATION_RESULT, 
                           MessageType.OCR_RESULT, MessageType.SEARCH_RESULT, MessageType.QUERY_RESPONSE]:
            await self._handle_response(message)
            
        elif message_type == MessageType.ERROR:
            await self._handle_error(message)
            
    async def _handle_response(self, message: Message):
        """Handle response messages from agents."""
        # Check if this is a response for an active workflow
        flow_id = message.header.route.flow_id
        if flow_id and flow_id in self.active_workflows:
            workflow = self.active_workflows[flow_id]
            
            # Record the step completion
            if workflow.current_step_index < len(workflow.steps):
                current_step = workflow.steps[workflow.current_step_index]
                current_step["completed_at"] = datetime.datetime.now().isoformat()
                current_step["result"] = {
                    "message_id": message.header.message_id,
                    "message_type": message.header.message_type,
                    "timestamp": message.header.timestamp
                }
                
            # Update workflow data with response
            if message.body:
                workflow.update_data(message.body)
                
            # Continue workflow execution
            await self._continue_workflow(workflow)
            
            # Notify subscribers
            await self._notify_workflow_subscribers(workflow.workflow_id, "step_completed", workflow)
            
        # Handle normal response correlation
        if message.header.correlation_id:
            await self.message_bus.handle_response(message)
            
    async def _handle_error(self, message: Message):
        """Handle error messages."""
        # Check if this is an error for an active workflow
        flow_id = message.header.route.flow_id
        if flow_id and flow_id in self.active_workflows:
            workflow = self.active_workflows[flow_id]
            
            # Record the error
            error_message = message.body.get("error_message", "Unknown error")
            error_code = message.body.get("error_code", "AGENT_ERROR")
            
            workflow.set_error(error_message, error_code, message.body)
            
            # Record the step failure
            if workflow.current_step_index < len(workflow.steps):
                current_step = workflow.steps[workflow.current_step_index]
                current_step["completed_at"] = datetime.datetime.now().isoformat()
                current_step["error"] = {
                    "message": error_message,
                    "code": error_code
                }
                
            # Metrics
            self.metrics_registry.counter("orchestration.workflows.failed").increment()
            self.metrics_registry.counter("orchestration.steps.failed").increment()
            
            # Notify subscribers
            await self._notify_workflow_subscribers(workflow.workflow_id, "workflow_failed", workflow)
            
            # Complete the workflow future if it exists
            if workflow.workflow_id in self.workflow_futures:
                future = self.workflow_futures[workflow.workflow_id]
                if not future.done():
                    future.set_result(workflow)
                    
        # Handle normal error correlation
        if message.header.correlation_id:
            await self.message_bus.handle_response(message)
            
    async def _handle_status_update(self, message: Message):
        """Handle status update messages."""
        agent_id = message.header.route.sender
        status = message.body.get("status")
        load = message.body.get("load")
        
        if agent_id and status:
            # Update agent status in the router
            self.message_bus.router.update_agent_status(
                agent_id,
                status,
                load
            )
            
    async def start_workflow(
        self,
        workflow_type: str,
        initial_data: Dict[str, Any],
        conversation_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[str, asyncio.Future]:
        """
        Start a new workflow of the specified type.
        
        Returns a tuple of (workflow_id, future) where the future will
        be completed when the workflow is done.
        """
        # Check if the workflow type is registered
        if not self.workflow_registry.has_workflow(workflow_type):
            raise ValueError(f"Unknown workflow type: {workflow_type}")
            
        # Create a new workflow context
        workflow = WorkflowContext(
            workflow_type=workflow_type,
            conversation_id=conversation_id,
            data=initial_data.copy(),
            metadata=metadata or {}
        )
        
        # Store the workflow
        self.active_workflows[workflow.workflow_id] = workflow
        
        # Create a future for tracking completion
        future = asyncio.Future()
        self.workflow_futures[workflow.workflow_id] = future
        
        # Start the workflow
        await self._execute_workflow(workflow)
        
        # Update metrics
        self.metrics_registry.counter("orchestration.workflows.total").increment()
        self.metrics_registry.gauge("orchestration.workflows.active").set(len(self.active_workflows))
        
        # Log workflow start
        logger.info(f"Started workflow {workflow.workflow_id} of type {workflow_type}")
        
        return workflow.workflow_id, future
        
    async def _execute_workflow(self, workflow: WorkflowContext):
        """Execute a workflow."""
        # Get the workflow definition
        workflow_def = self.workflow_registry.get_workflow(workflow.workflow_type)
        if not workflow_def:
            workflow.set_error(f"Workflow type not found: {workflow.workflow_type}")
            return
            
        # Update status
        workflow.update_status(WorkflowStatus.RUNNING)
        
        # Create a trace span for the workflow
        with self.tracer.span(
            f"workflow.{workflow.workflow_type}",
            metadata={"workflow_id": workflow.workflow_id}
        ) as span:
            # Get initial steps
            try:
                steps = await workflow_def.get_initial_steps(workflow.data)
                
                # Add steps to the workflow
                for step in steps:
                    workflow.add_step(step)
                    
                # Start execution
                await self._continue_workflow(workflow)
                
                # Notify subscribers
                await self._notify_workflow_subscribers(workflow.workflow_id, "workflow_started", workflow)
                
            except Exception as e:
                logger.error(f"Error starting workflow {workflow.workflow_id}: {str(e)}")
                workflow.set_error(f"Error starting workflow: {str(e)}")
                
                # Metrics
                self.metrics_registry.counter("orchestration.workflows.failed").increment()
                
                # Notify subscribers
                await self._notify_workflow_subscribers(workflow.workflow_id, "workflow_failed", workflow)
                
                # Complete the future
                if workflow.workflow_id in self.workflow_futures:
                    future = self.workflow_futures[workflow.workflow_id]
                    if not future.done():
                        future.set_result(workflow)
                        
    async def _continue_workflow(self, workflow: WorkflowContext):
        """Continue execution of a workflow."""
        # Check if the workflow is already done
        if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED]:
            return
            
        # Get the workflow definition
        workflow_def = self.workflow_registry.get_workflow(workflow.workflow_type)
        if not workflow_def:
            workflow.set_error(f"Workflow definition not found: {workflow.workflow_type}")
            return
            
        # Check if we're at the end of the steps
        if workflow.current_step_index >= len(workflow.steps):
            # Determine next steps
            try:
                next_steps = await workflow_def.determine_next_steps(workflow.data, workflow.steps)
                
                # If there are no next steps, the workflow is complete
                if not next_steps:
                    workflow.update_status(WorkflowStatus.COMPLETED)
                    
                    # Calculate duration
                    created_time = datetime.datetime.fromisoformat(workflow.created_at)
                    completed_time = datetime.datetime.fromisoformat(workflow.completed_at)
                    duration_ms = (completed_time - created_time).total_seconds() * 1000
                    
                    # Record metrics
                    self.metrics_registry.counter("orchestration.workflows.completed").increment()
                    self.metrics_registry.gauge("orchestration.workflows.active").set(len(self.active_workflows) - 1)
                    self.metrics_registry.histogram("orchestration.workflow.duration").observe(duration_ms)
                    
                    # Notify subscribers
                    await self._notify_workflow_subscribers(workflow.workflow_id, "workflow_completed", workflow)
                    
                    # Complete the future
                    if workflow.workflow_id in self.workflow_futures:
                        future = self.workflow_futures[workflow.workflow_id]
                        if not future.done():
                            future.set_result(workflow)
                            
                    return
                    
                # Add next steps and continue execution
                for step in next_steps:
                    workflow.add_step(step)
                    
            except Exception as e:
                logger.error(f"Error determining next steps for workflow {workflow.workflow_id}: {str(e)}")
                workflow.set_error(f"Error determining next steps: {str(e)}")
                
                # Metrics
                self.metrics_registry.counter("orchestration.workflows.failed").increment()
                self.metrics_registry.gauge("orchestration.workflows.active").set(len(self.active_workflows) - 1)
                
                # Notify subscribers
                await self._notify_workflow_subscribers(workflow.workflow_id, "workflow_failed", workflow)
                
                # Complete the future
                if workflow.workflow_id in self.workflow_futures:
                    future = self.workflow_futures[workflow.workflow_id]
                    if not future.done():
                        future.set_result(workflow)
                        
                return
                
        # Execute the current step
        await self._execute_step(workflow)
        
    async def _execute_step(self, workflow: WorkflowContext):
        """Execute a single workflow step."""
        # Get the current step
        if workflow.current_step_index >= len(workflow.steps):
            return
            
        current_step = workflow.steps[workflow.current_step_index]
        
        # Mark step as started
        current_step["started_at"] = datetime.datetime.now().isoformat()
        
        # Create a trace span for the step
        with self.tracer.span(
            f"workflow_step.{current_step.get('type', 'unknown')}",
            trace_id=workflow.workflow_id,
            metadata={"step_index": workflow.current_step_index}
        ) as span:
            # Get the agent for this step
            agent_id = current_step.get("agent")
            if not agent_id:
                # Try to find an agent with the required capability
                capability = current_step.get("capability")
                if capability:
                    agent_id = self.message_bus.router.get_optimal_agent(capability)
                    
            if not agent_id:
                logger.error(f"No agent found for step: {current_step}")
                workflow.set_error(f"No agent found for step: {current_step.get('type', 'unknown')}")
                return
                
            # Prepare the message
            step_type = current_step.get("type", "unknown")
            message_type = self._get_message_type_for_step(step_type)
            
            # Create the message body from the step parameters
            body = current_step.get("parameters", {}).copy()
            
            # Add any additional context from the workflow data
            if "context_keys" in current_step:
                for key in current_step["context_keys"]:
                    if key in workflow.data:
                        body[key] = workflow.data[key]
                        
            # Create the message
            message = create_message(
                message_type=message_type,
                sender="orchestration_manager",
                recipient=agent_id,
                body=body,
                flow_id=workflow.workflow_id,
                correlation_id=str(uuid.uuid4())
            )
            
            # Record metrics
            self.metrics_registry.counter("orchestration.steps.total").increment()
            
            # Send the message
            try:
                success = await self.message_bus.send_message(message)
                if not success:
                    workflow.set_error(f"Failed to send message to agent: {agent_id}")
                    # Increment the step index to move past this step
                    workflow.current_step_index += 1
                    # Continue the workflow (will trigger error handling)
                    await self._continue_workflow(workflow)
                    
            except Exception as e:
                logger.error(f"Error executing step {workflow.current_step_index} of workflow {workflow.workflow_id}: {str(e)}")
                workflow.set_error(f"Error executing step: {str(e)}")
                # Increment the step index to move past this step
                workflow.current_step_index += 1
                # Continue the workflow (will trigger error handling)
                await self._continue_workflow(workflow)
                
    def _get_message_type_for_step(self, step_type: str) -> MessageType:
        """Get the appropriate message type for a step type."""
        step_type_mapping = {
            "computation": MessageType.COMPUTATION_REQUEST,
            "visualization": MessageType.VISUALIZATION_REQUEST,
            "ocr": MessageType.OCR_REQUEST,
            "search": MessageType.SEARCH_REQUEST,
            "query": MessageType.QUERY
        }
        
        return step_type_mapping.get(step_type, MessageType.QUERY)
        
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get a workflow by ID."""
        return self.active_workflows.get(workflow_id)
        
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id not in self.active_workflows:
            return False
            
        workflow = self.active_workflows[workflow_id]
        if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED]:
            return False
            
        workflow.update_status(WorkflowStatus.CANCELED)
        
        # Notify subscribers
        await self._notify_workflow_subscribers(workflow_id, "workflow_canceled", workflow)
        
        # Complete the future
        if workflow_id in self.workflow_futures:
            future = self.workflow_futures[workflow_id]
            if not future.done():
                future.set_result(workflow)
                
        # Update metrics
        self.metrics_registry.gauge("orchestration.workflows.active").set(len(self.active_workflows) - 1)
        
        return True
        
    async def cleanup_completed_workflows(self, max_age_seconds: int = 3600):
        """Clean up completed workflows that are older than the specified age."""
        now = datetime.datetime.now()
        workflows_to_remove = []
        
        for workflow_id, workflow in self.active_workflows.items():
            if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED]:
                if workflow.completed_at:
                    completed_time = datetime.datetime.fromisoformat(workflow.completed_at)
                    age_seconds = (now - completed_time).total_seconds()
                    
                    if age_seconds > max_age_seconds:
                        workflows_to_remove.append(workflow_id)
                        
        # Remove the old workflows
        for workflow_id in workflows_to_remove:
            del self.active_workflows[workflow_id]
            
            # Clean up futures
            if workflow_id in self.workflow_futures:
                del self.workflow_futures[workflow_id]
                
            # Clean up subscriptions
            if workflow_id in self.workflow_subscription_callbacks:
                del self.workflow_subscription_callbacks[workflow_id]
                
        if workflows_to_remove:
            logger.info(f"Cleaned up {len(workflows_to_remove)} completed workflows")
            
    def subscribe_to_workflow(self, workflow_id: str, callback: Callable):
        """Subscribe to workflow events."""
        if workflow_id not in self.workflow_subscription_callbacks:
            self.workflow_subscription_callbacks[workflow_id] = []
            
        self.workflow_subscription_callbacks[workflow_id].append(callback)
        
    def unsubscribe_from_workflow(self, workflow_id: str, callback: Callable):
        """Unsubscribe from workflow events."""
        if workflow_id in self.workflow_subscription_callbacks:
            if callback in self.workflow_subscription_callbacks[workflow_id]:
                self.workflow_subscription_callbacks[workflow_id].remove(callback)
                
    async def _notify_workflow_subscribers(self, workflow_id: str, event_type: str, workflow: WorkflowContext):
        """Notify subscribers of workflow events."""
        if workflow_id not in self.workflow_subscription_callbacks:
            return
            
        event_data = {
            "event_type": event_type,
            "workflow_id": workflow_id,
            "workflow": workflow.dict(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        callbacks = self.workflow_subscription_callbacks[workflow_id].copy()
        for callback in callbacks:
            try:
                await callback(event_data)
            except Exception as e:
                logger.error(f"Error in workflow subscription callback: {str(e)}")


# Create a singleton instance
_orchestration_manager_instance = None

def get_orchestration_manager() -> OrchestrationManager:
    """Get or create the orchestration manager singleton instance."""
    global _orchestration_manager_instance
    if _orchestration_manager_instance is None:
        _orchestration_manager_instance = OrchestrationManager()
    return _orchestration_manager_instance

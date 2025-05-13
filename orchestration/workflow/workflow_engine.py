"""
Workflow Engine for the Mathematical Multimodal LLM System.

This module provides a workflow engine that handles state persistence,
long-running operations, and error recovery for complex workflows.
"""
import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import logging
import datetime
import copy
import traceback
import time

from ..message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message, create_error_response
)
from ..message_bus.rabbitmq_wrapper import get_message_bus
from ..monitoring.logger import get_logger, set_correlation_id
from ..monitoring.tracing import get_tracer, Span
from ..monitoring.metrics import get_registry
from ..agents.registry import get_agent_registry
from ..agents.load_balancer import get_load_balancer
from .workflow_registry import get_workflow_registry, WorkflowDefinition

logger = get_logger(__name__)


class WorkflowExecutionStatus:
    """Status values for workflow executions."""
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"


class ActivityStatus:
    """Status values for workflow activities."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"


class WorkflowExecution:
    """
    Represents a single execution of a workflow.
    
    A workflow execution contains the complete state of a workflow,
    including all activities, their status, and the workflow context.
    """
    
    def __init__(
        self,
        workflow_type: str,
        workflow_id: Optional[str] = None,
        initial_context: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a workflow execution.
        
        Args:
            workflow_type: The type of workflow to execute
            workflow_id: Optional custom workflow ID
            initial_context: Initial context for the workflow
            metadata: Additional metadata for the workflow
        """
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.workflow_type = workflow_type
        self.status = WorkflowExecutionStatus.CREATED
        self.activities: List[Dict[str, Any]] = []
        self.current_activity_index = -1
        self.context = initial_context or {}
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = self.created_at
        self.completed_at = None
        self.error = None
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.checkpoint_interval_seconds = 30  # Save state every 30 seconds
        self.last_checkpoint_time = time.time()
        
    def update_status(self, status: str):
        """
        Update the status of the workflow execution.
        
        Args:
            status: New status
        """
        self.status = status
        self.updated_at = datetime.datetime.now().isoformat()
        
        if status in [
            WorkflowExecutionStatus.COMPLETED,
            WorkflowExecutionStatus.FAILED,
            WorkflowExecutionStatus.CANCELED,
            WorkflowExecutionStatus.TIMED_OUT
        ]:
            self.completed_at = self.updated_at
            
    def add_activity(self, activity: Dict[str, Any]):
        """
        Add an activity to the workflow execution.
        
        Args:
            activity: Activity definition
        """
        # Ensure the activity has required fields
        if "id" not in activity:
            activity["id"] = str(uuid.uuid4())
            
        if "status" not in activity:
            activity["status"] = ActivityStatus.PENDING
            
        if "created_at" not in activity:
            activity["created_at"] = datetime.datetime.now().isoformat()
            
        if "updated_at" not in activity:
            activity["updated_at"] = activity["created_at"]
            
        # Add the activity
        self.activities.append(activity)
        self.updated_at = datetime.datetime.now().isoformat()
        
    def set_error(self, error_message: str, error_code: str = "WORKFLOW_ERROR", details: Dict[str, Any] = None):
        """
        Set an error on the workflow execution.
        
        Args:
            error_message: Error message
            error_code: Error code
            details: Additional error details
        """
        self.error = {
            "message": error_message,
            "code": error_code,
            "details": details or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.update_status(WorkflowExecutionStatus.FAILED)
        
    def checkpoint(self):
        """
        Create a checkpoint of the current workflow state.
        
        This method saves the current state of the workflow to the checkpoint history,
        enabling recovery in case of failures.
        """
        # Create a deep copy of the current state
        state = {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "current_activity_index": self.current_activity_index,
            "context": copy.deepcopy(self.context),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to checkpoint history
        self.checkpoint_history.append(state)
        self.last_checkpoint_time = time.time()
        
        # Keep only the last 10 checkpoints to save memory
        if len(self.checkpoint_history) > 10:
            self.checkpoint_history = self.checkpoint_history[-10:]
            
    def restore_checkpoint(self, checkpoint_index: int = -1) -> bool:
        """
        Restore the workflow state from a checkpoint.
        
        Args:
            checkpoint_index: Index of the checkpoint to restore (-1 for latest)
            
        Returns:
            True if restored successfully, False otherwise
        """
        if not self.checkpoint_history:
            return False
            
        # Validate index
        if checkpoint_index < -len(self.checkpoint_history) or checkpoint_index >= len(self.checkpoint_history):
            return False
            
        # Get the checkpoint
        checkpoint = self.checkpoint_history[checkpoint_index]
        
        # Restore state
        self.status = checkpoint["status"]
        self.current_activity_index = checkpoint["current_activity_index"]
        self.context = copy.deepcopy(checkpoint["context"])
        self.updated_at = datetime.datetime.now().isoformat()
        
        return True
        
    def set_activity_status(self, activity_index: int, status: str, result: Dict[str, Any] = None, error: Dict[str, Any] = None):
        """
        Update the status of an activity.
        
        Args:
            activity_index: Index of the activity
            status: New status
            result: Optional result data
            error: Optional error data
        """
        if activity_index < 0 or activity_index >= len(self.activities):
            return
            
        # Update the activity
        activity = self.activities[activity_index]
        activity["status"] = status
        activity["updated_at"] = datetime.datetime.now().isoformat()
        
        if result is not None:
            activity["result"] = result
            
        if error is not None:
            activity["error"] = error
            
        # Update workflow timestamp
        self.updated_at = datetime.datetime.now().isoformat()
        
    def should_checkpoint(self) -> bool:
        """
        Check if a checkpoint should be created.
        
        Returns:
            True if a checkpoint should be created, False otherwise
        """
        # Check if enough time has passed since the last checkpoint
        elapsed_seconds = time.time() - self.last_checkpoint_time
        return elapsed_seconds >= self.checkpoint_interval_seconds
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow execution to a dictionary.
        
        Returns:
            Dictionary representation of the workflow execution
        """
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "status": self.status,
            "activities": self.activities,
            "current_activity_index": self.current_activity_index,
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowExecution':
        """
        Create a workflow execution from a dictionary.
        
        Args:
            data: Dictionary representation of a workflow execution
            
        Returns:
            WorkflowExecution instance
        """
        workflow = cls(
            workflow_type=data["workflow_type"],
            workflow_id=data["workflow_id"],
            initial_context=data.get("context", {}),
            metadata=data.get("metadata", {})
        )
        
        workflow.status = data["status"]
        workflow.activities = data.get("activities", [])
        workflow.current_activity_index = data.get("current_activity_index", -1)
        workflow.created_at = data.get("created_at", workflow.created_at)
        workflow.updated_at = data.get("updated_at", workflow.updated_at)
        workflow.completed_at = data.get("completed_at")
        workflow.error = data.get("error")
        
        return workflow


class ActivityExecutionContext:
    """
    Context for executing a single activity.
    
    This class provides a context for executing a workflow activity,
    including methods for managing state and handling errors.
    """
    
    def __init__(
        self,
        workflow_execution: WorkflowExecution,
        activity_index: int,
        activity: Dict[str, Any],
        engine: 'WorkflowEngine'
    ):
        """
        Initialize an activity execution context.
        
        Args:
            workflow_execution: The parent workflow execution
            activity_index: The index of the activity in the workflow
            activity: The activity definition
            engine: The workflow engine
        """
        self.workflow_execution = workflow_execution
        self.activity_index = activity_index
        self.activity = activity
        self.engine = engine
        self.context = workflow_execution.context
        
    async def execute(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the activity.
        
        Returns:
            Tuple of (success, result)
        """
        # Update activity status
        self.workflow_execution.set_activity_status(self.activity_index, ActivityStatus.RUNNING)
        
        # Get the agent for this activity
        agent_id = self.activity.get("agent")
        capability = self.activity.get("capability")
        
        if not agent_id and capability:
            # Try to find an agent with the required capability
            load_balancer = get_load_balancer()
            agent_id = load_balancer.select_agent(capability)
            
        if not agent_id:
            error = {
                "message": f"No agent found for capability: {capability}",
                "code": "AGENT_NOT_FOUND",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.workflow_execution.set_activity_status(
                self.activity_index,
                ActivityStatus.FAILED,
                error=error
            )
            
            return False, {"error": error}
            
        # Create the message for the agent
        activity_type = self.activity.get("type", "unknown")
        message_type = self._get_message_type_for_activity(activity_type)
        
        # Create message body from activity parameters and context
        body = self.activity.get("parameters", {}).copy()
        
        # Add context keys to the body if specified
        context_keys = self.activity.get("context_keys", [])
        for key in context_keys:
            if key in self.context:
                body[key] = self.context[key]
                
        try:
            # Send the message to the agent
            message = create_message(
                message_type=message_type,
                sender="workflow_engine",
                recipient=agent_id,
                body=body,
                flow_id=self.workflow_execution.workflow_id,
                correlation_id=self.activity.get("id")
            )
            
            message_bus = get_message_bus()
            sent = await message_bus.send_message(message)
            
            if not sent:
                error = {
                    "message": f"Failed to send message to agent: {agent_id}",
                    "code": "MESSAGE_SEND_FAILED",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                self.workflow_execution.set_activity_status(
                    self.activity_index,
                    ActivityStatus.FAILED,
                    error=error
                )
                
                return False, {"error": error}
                
            # Create a future for the response
            response_future = asyncio.Future()
            self.engine.register_response_future(self.activity.get("id"), response_future)
            
            # Wait for the response (timeout handled by the engine)
            response = await response_future
            
            # Handle the response
            if response.header.message_type == MessageType.ERROR:
                # Agent returned an error
                error = {
                    "message": response.body.get("error_message", "Unknown error"),
                    "code": response.body.get("error_code", "AGENT_ERROR"),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                self.workflow_execution.set_activity_status(
                    self.activity_index,
                    ActivityStatus.FAILED,
                    error=error
                )
                
                return False, {"error": error}
                
            # Success - update activity status and workflow context
            self.workflow_execution.set_activity_status(
                self.activity_index,
                ActivityStatus.COMPLETED,
                result=response.body
            )
            
            # Update context with response body
            self.context.update(response.body)
            
            return True, response.body
            
        except asyncio.CancelledError:
            # Activity was canceled
            error = {
                "message": "Activity execution was canceled",
                "code": "ACTIVITY_CANCELED",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.workflow_execution.set_activity_status(
                self.activity_index,
                ActivityStatus.CANCELED,
                error=error
            )
            
            return False, {"error": error}
            
        except asyncio.TimeoutError:
            # Activity timed out
            error = {
                "message": "Activity execution timed out",
                "code": "ACTIVITY_TIMEOUT",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.workflow_execution.set_activity_status(
                self.activity_index,
                ActivityStatus.TIMED_OUT,
                error=error
            )
            
            return False, {"error": error}
            
        except Exception as e:
            # Other error
            error = {
                "message": f"Error executing activity: {str(e)}",
                "code": "ACTIVITY_ERROR",
                "exception": traceback.format_exc(),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.workflow_execution.set_activity_status(
                self.activity_index,
                ActivityStatus.FAILED,
                error=error
            )
            
            return False, {"error": error}
            
    def _get_message_type_for_activity(self, activity_type: str) -> MessageType:
        """
        Get the appropriate message type for an activity type.
        
        Args:
            activity_type: Type of activity
            
        Returns:
            Appropriate message type
        """
        activity_type_mapping = {
            "computation": MessageType.COMPUTATION_REQUEST,
            "visualization": MessageType.VISUALIZATION_REQUEST,
            "ocr": MessageType.OCR_REQUEST,
            "search": MessageType.SEARCH_REQUEST,
            "query": MessageType.QUERY
        }
        
        return activity_type_mapping.get(activity_type, MessageType.QUERY)


class WorkflowEngine:
    """
    Engine for executing workflows.
    
    This class provides functionality for executing workflows, managing
    their state, and handling errors and recovery.
    """
    
    def __init__(self):
        """Initialize the workflow engine."""
        self.workflow_registry = get_workflow_registry()
        self.message_bus = get_message_bus()
        self.agent_registry = get_agent_registry()
        self.tracer = get_tracer()
        self.metrics = get_registry()
        
        # Active workflow executions
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        
        # Response futures for activity completion
        self.response_futures: Dict[str, asyncio.Future] = {}
        
        # Workflow completion futures
        self.completion_futures: Dict[str, asyncio.Future] = {}
        
        # Workflow event subscribers
        self.event_subscribers: Dict[str, List[Callable]] = {}
        
        # Default activity timeout (seconds)
        self.default_activity_timeout = 60.0
        
        # Initialize metrics
        self._setup_metrics()
        
    async def start(self):
        """Start the workflow engine."""
        logger.info("Starting workflow engine")
        
        # Connect to the message bus if not already connected
        if not self.message_bus.connection:
            await self.message_bus.connect()
            
        # Setup listener for response messages
        await self.message_bus.add_message_listener(
            MessageType.COMPUTATION_RESULT,
            self._handle_response
        )
        
        await self.message_bus.add_message_listener(
            MessageType.VISUALIZATION_RESULT,
            self._handle_response
        )
        
        await self.message_bus.add_message_listener(
            MessageType.OCR_RESULT,
            self._handle_response
        )
        
        await self.message_bus.add_message_listener(
            MessageType.SEARCH_RESULT,
            self._handle_response
        )
        
        await self.message_bus.add_message_listener(
            MessageType.QUERY_RESPONSE,
            self._handle_response
        )
        
        await self.message_bus.add_message_listener(
            MessageType.ERROR,
            self._handle_response
        )
        
        # Start background tasks
        self._checkpoint_task = asyncio.create_task(self._checkpoint_workflows_periodically())
        self._cleanup_task = asyncio.create_task(self._cleanup_workflows_periodically())
        
        logger.info("Workflow engine started")
        
    async def stop(self):
        """Stop the workflow engine."""
        logger.info("Stopping workflow engine")
        
        # Cancel background tasks
        if hasattr(self, '_checkpoint_task'):
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
                
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Cancel all active workflow executions
        for workflow_id, workflow in list(self.active_workflows.items()):
            await self.cancel_workflow(workflow_id)
            
        # Clear futures
        for future in self.response_futures.values():
            if not future.done():
                future.cancel()
                
        for future in self.completion_futures.values():
            if not future.done():
                future.cancel()
                
        logger.info("Workflow engine stopped")
        
    def _setup_metrics(self):
        """Setup workflow metrics."""
        registry = self.metrics
        
        # Workflow metrics
        registry.counter("workflow.executions.total", "Total number of workflow executions")
        registry.counter("workflow.executions.completed", "Number of completed workflows")
        registry.counter("workflow.executions.failed", "Number of failed workflows")
        registry.gauge("workflow.executions.active", "Number of active workflows")
        
        # Activity metrics
        registry.counter("workflow.activities.total", "Total number of workflow activities")
        registry.counter("workflow.activities.completed", "Number of completed activities")
        registry.counter("workflow.activities.failed", "Number of failed activities")
        registry.histogram(
            "workflow.activities.duration",
            "Activity execution duration (ms)",
            buckets=[100, 500, 1000, 2500, 5000, 10000, 30000, 60000]
        )
        
        # Workflow duration metric
        registry.histogram(
            "workflow.execution.duration",
            "Workflow execution duration (ms)",
            buckets=[1000, 5000, 10000, 30000, 60000, 300000, 600000]
        )
        
    async def execute_workflow(
        self,
        workflow_type: str,
        initial_context: Dict[str, Any],
        workflow_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        wait_for_completion: bool = False,
        timeout: Optional[float] = None
    ) -> Tuple[str, Optional[WorkflowExecution]]:
        """
        Execute a workflow.
        
        Args:
            workflow_type: Type of workflow to execute
            initial_context: Initial context for the workflow
            workflow_id: Optional custom workflow ID
            metadata: Additional metadata for the workflow
            wait_for_completion: Whether to wait for workflow completion
            timeout: Optional timeout for waiting (seconds)
            
        Returns:
            Tuple of (workflow_id, workflow_execution if wait_for_completion else None)
        """
        # Check if the workflow type is registered
        if not self.workflow_registry.has_workflow(workflow_type):
            raise ValueError(f"Unknown workflow type: {workflow_type}")
            
        # Create a workflow execution
        workflow = WorkflowExecution(
            workflow_type=workflow_type,
            workflow_id=workflow_id,
            initial_context=initial_context,
            metadata=metadata
        )
        
        # Register the workflow
        self.active_workflows[workflow.workflow_id] = workflow
        
        # Record metrics
        self.metrics.counter("workflow.executions.total").increment()
        self.metrics.gauge("workflow.executions.active").set(len(self.active_workflows))
        
        # Start the workflow
        await self._execute_workflow(workflow)
        
        if wait_for_completion:
            # Wait for workflow completion
            completion_future = asyncio.Future()
            self.completion_futures[workflow.workflow_id] = completion_future
            
            try:
                await asyncio.wait_for(completion_future, timeout=timeout)
                return workflow.workflow_id, workflow
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for workflow {workflow.workflow_id} to complete")
                
                # Remove the future but keep the workflow running
                del self.completion_futures[workflow.workflow_id]
                
                return workflow.workflow_id, None
                
        return workflow.workflow_id, None
        
    async def _execute_workflow(self, workflow: WorkflowExecution):
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow execution to run
        """
        # Check if the workflow is already completed
        if workflow.status in [
            WorkflowExecutionStatus.COMPLETED,
            WorkflowExecutionStatus.FAILED,
            WorkflowExecutionStatus.CANCELED,
            WorkflowExecutionStatus.TIMED_OUT
        ]:
            logger.warning(f"Workflow {workflow.workflow_id} is already in terminal state: {workflow.status}")
            return
            
        # Set the workflow status to running
        workflow.update_status(WorkflowExecutionStatus.RUNNING)
        
        # Emit workflow started event
        await self._emit_workflow_event(workflow.workflow_id, "workflow_started", workflow)
        
        # Get the workflow definition
        workflow_def = self.workflow_registry.get_workflow(workflow.workflow_type)
        if not workflow_def:
            workflow.set_error(f"Workflow type not found: {workflow.workflow_type}")
            return
            
        # Create a trace span for the workflow
        with self.tracer.span(
            f"workflow.{workflow.workflow_type}",
            metadata={"workflow_id": workflow.workflow_id}
        ) as span:
            # Start time for duration calculation
            start_time = time.time()
            
            try:
                # Determine initial steps if needed
                if not workflow.activities:
                    # Get initial steps
                    initial_steps = await workflow_def.get_initial_steps(workflow.context)
                    
                    # Add activities
                    for step in initial_steps:
                        workflow.add_activity(step)
                        
                # Execute activities
                await self._continue_workflow(workflow, workflow_def)
                
                # Calculate duration if the workflow is complete
                if workflow.status in [
                    WorkflowExecutionStatus.COMPLETED,
                    WorkflowExecutionStatus.FAILED,
                    WorkflowExecutionStatus.CANCELED,
                    WorkflowExecutionStatus.TIMED_OUT
                ]:
                    duration_ms = (time.time() - start_time) * 1000
                    self.metrics.histogram("workflow.execution.duration").observe(duration_ms)
                    
            except Exception as e:
                logger.error(f"Error executing workflow {workflow.workflow_id}: {str(e)}")
                traceback.print_exc()
                
                # Set workflow error
                workflow.set_error(
                    error_message=f"Error executing workflow: {str(e)}",
                    error_code="WORKFLOW_ERROR",
                    details={"exception": traceback.format_exc()}
                )
                
                # Emit workflow failed event
                await self._emit_workflow_event(workflow.workflow_id, "workflow_failed", workflow)
                
                # Complete the future if one exists
                if workflow.workflow_id in self.completion_futures:
                    future = self.completion_futures[workflow.workflow_id]
                    if not future.done():
                        future.set_result(workflow)
                        
                # Record failure metrics
                self.metrics.counter("workflow.executions.failed").increment()
                
    async def _continue_workflow(self, workflow: WorkflowExecution, workflow_def: WorkflowDefinition):
        """
        Continue executing a workflow.
        
        Args:
            workflow: Workflow execution to continue
            workflow_def: Workflow definition
        """
        # Check if the workflow is already completed
        if workflow.status in [
            WorkflowExecutionStatus.COMPLETED,
            WorkflowExecutionStatus.FAILED,
            WorkflowExecutionStatus.CANCELED,
            WorkflowExecutionStatus.TIMED_OUT
        ]:
            return
            
        # Check if we need to execute more activities
        if workflow.current_activity_index < len(workflow.activities) - 1:
            # Move to the next activity
            workflow.current_activity_index += 1
            activity_index = workflow.current_activity_index
            
            # Execute the current activity
            if activity_index < len(workflow.activities):
                activity = workflow.activities[activity_index]
                
                # Check if the activity is already completed
                if activity.get("status") == ActivityStatus.COMPLETED:
                    # Skip this activity and continue
                    await self._continue_workflow(workflow, workflow_def)
                    return
                    
                # Execute the activity
                activity_context = ActivityExecutionContext(
                    workflow_execution=workflow,
                    activity_index=activity_index,
                    activity=activity,
                    engine=self
                )
                
                # Record activity metrics
                self.metrics.counter("workflow.activities.total").increment()
                
                # Execute the activity with timeout
                start_time = time.time()
                activity_id = activity.get("id", "unknown")
                
                with self.tracer.span(
                    f"workflow_activity.{activity.get('name', 'unknown')}",
                    trace_id=workflow.workflow_id,
                    metadata={"activity_id": activity_id}
                ):
                    success, result = await activity_context.execute()
                    
                # Calculate activity duration
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.histogram("workflow.activities.duration").observe(duration_ms)
                
                # Update metrics based on success
                if success:
                    self.metrics.counter("workflow.activities.completed").increment()
                else:
                    self.metrics.counter("workflow.activities.failed").increment()
                    
                # Emit activity completed event
                await self._emit_workflow_event(
                    workflow.workflow_id,
                    "activity_completed" if success else "activity_failed",
                    {
                        "workflow": workflow,
                        "activity_index": activity_index,
                        "activity": activity,
                        "success": success,
                        "result": result
                    }
                )
                
                # Check if we should create a checkpoint
                if workflow.should_checkpoint():
                    workflow.checkpoint()
                    
                # Check for activity failure
                if not success:
                    # Check if this activity has recovery options
                    recovery_options = activity.get("recovery_options", {})
                    
                    if recovery_options:
                        # Check if retries are available
                        max_retries = recovery_options.get("max_retries", 0)
                        current_attempts = activity.get("attempts", 0) + 1
                        activity["attempts"] = current_attempts
                        
                        if current_attempts <= max_retries:
                            # Retry the activity
                            logger.info(f"Retrying activity {activity_id} (attempt {current_attempts}/{max_retries})")
                            
                            # Reset activity status
                            workflow.set_activity_status(activity_index, ActivityStatus.PENDING)
                            
                            # Stay on the same activity index for retry
                            workflow.current_activity_index -= 1
                            
                            # Continue workflow (will retry this activity)
                            await self._continue_workflow(workflow, workflow_def)
                            return
                            
                        # Check for fallback
                        fallback = recovery_options.get("fallback")
                        
                        if fallback:
                            logger.info(f"Using fallback {fallback} for failed activity {activity_id}")
                            
                            # Record fallback in context
                            if "recovery" not in workflow.context:
                                workflow.context["recovery"] = {
                                    "fallbacks": {},
                                    "errors": []
                                }
                                
                            workflow.context["recovery"]["fallbacks"] = {
                                **workflow.context["recovery"].get("fallbacks", {}),
                                activity.get("name", "unknown"): fallback
                            }
                            
                            workflow.context["recovery"]["errors"] = [
                                *workflow.context["recovery"].get("errors", []),
                                {
                                    "activity": activity.get("name", "unknown"),
                                    "error": result.get("error", {"message": "Unknown error"}),
                                    "fallback": fallback,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                            ]
                            
                            # Continue workflow (will get new steps)
                            await self._get_next_steps(workflow, workflow_def)
                            return
                            
                    # No recovery options or all exhausted
                    logger.error(f"Activity {activity_id} failed with no recovery options")
                    
                    # Set workflow as failed
                    workflow.set_error(
                        error_message=f"Activity failed: {result.get('error', {}).get('message', 'Unknown error')}",
                        error_code="ACTIVITY_FAILED",
                        details=result.get("error")
                    )
                    
                    # Emit workflow failed event
                    await self._emit_workflow_event(workflow.workflow_id, "workflow_failed", workflow)
                    
                    # Complete the future if one exists
                    if workflow.workflow_id in self.completion_futures:
                        future = self.completion_futures[workflow.workflow_id]
                        if not future.done():
                            future.set_result(workflow)
                            
                    # Record failure metrics
                    self.metrics.counter("workflow.executions.failed").increment()
                    
                    return
                    
                # Continue workflow execution
                await self._continue_workflow(workflow, workflow_def)
                
        else:
            # All current activities are completed, get next steps
            await self._get_next_steps(workflow, workflow_def)
            
    async def _get_next_steps(self, workflow: WorkflowExecution, workflow_def: WorkflowDefinition):
        """
        Get the next steps for a workflow.
        
        Args:
            workflow: Workflow execution
            workflow_def: Workflow definition
        """
        # Determine the next steps
        try:
            # Extract completed steps for the workflow definition
            completed_steps = []
            for activity in workflow.activities:
                if activity.get("status") == ActivityStatus.COMPLETED:
                    step = {
                        "name": activity.get("name"),
                        "type": activity.get("type"),
                        "result": activity.get("result", {})
                    }
                    completed_steps.append(step)
                    
            next_steps = await workflow_def.determine_next_steps(workflow.context, completed_steps)
            
            if not next_steps:
                # No more steps, workflow is complete
                workflow.update_status(WorkflowExecutionStatus.COMPLETED)
                
                # Emit workflow completed event
                await self._emit_workflow_event(workflow.workflow_id, "workflow_completed", workflow)
                
                # Complete the future if one exists
                if workflow.workflow_id in self.completion_futures:
                    future = self.completion_futures[workflow.workflow_id]
                    if not future.done():
                        future.set_result(workflow)
                        
                # Record completion metrics
                self.metrics.counter("workflow.executions.completed").increment()
                self.metrics.gauge("workflow.executions.active").set(len(self.active_workflows) - 1)
                
                return
                
            # Add the new steps as activities
            for step in next_steps:
                workflow.add_activity(step)
                
            # Continue workflow execution
            await self._continue_workflow(workflow, workflow_def)
            
        except Exception as e:
            logger.error(f"Error determining next steps for workflow {workflow.workflow_id}: {str(e)}")
            traceback.print_exc()
            
            # Set workflow error
            workflow.set_error(
                error_message=f"Error determining next steps: {str(e)}",
                error_code="NEXT_STEPS_ERROR",
                details={"exception": traceback.format_exc()}
            )
            
            # Emit workflow failed event
            await self._emit_workflow_event(workflow.workflow_id, "workflow_failed", workflow)
            
            # Complete the future if one exists
            if workflow.workflow_id in self.completion_futures:
                future = self.completion_futures[workflow.workflow_id]
                if not future.done():
                    future.set_result(workflow)
                    
            # Record failure metrics
            self.metrics.counter("workflow.executions.failed").increment()
            
    async def _handle_response(self, message: Message):
        """
        Handle a response message from an agent.
        
        Args:
            message: Response message
        """
        # Check if this is a response for an active workflow
        flow_id = message.header.route.flow_id
        correlation_id = message.header.correlation_id
        
        if flow_id and flow_id in self.active_workflows:
            # This is a workflow-related message
            logger.debug(f"Received response message for workflow {flow_id}, correlation {correlation_id}")
            
            # Find the future for this correlation ID
            if correlation_id in self.response_futures:
                future = self.response_futures[correlation_id]
                
                if not future.done():
                    future.set_result(message)
                    
                # Remove the future
                del self.response_futures[correlation_id]
                
        else:
            # Not a workflow-related message, ignore
            pass
            
    def register_response_future(self, correlation_id: str, future: asyncio.Future):
        """
        Register a future for a response message.
        
        Args:
            correlation_id: Correlation ID for the message
            future: Future to complete when the response is received
        """
        self.response_futures[correlation_id] = future
        
        # Set up a timeout for the future
        asyncio.create_task(self._timeout_future(correlation_id, future, self.default_activity_timeout))
        
    async def _timeout_future(self, correlation_id: str, future: asyncio.Future, timeout: float):
        """
        Time out a future after a specified duration.
        
        Args:
            correlation_id: Correlation ID for the message
            future: Future to time out
            timeout: Timeout in seconds
        """
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            # Future timed out
            if correlation_id in self.response_futures and not future.done():
                logger.warning(f"Response timed out for correlation ID {correlation_id}")
                future.set_exception(asyncio.TimeoutError(f"Response timed out for {correlation_id}"))
                del self.response_futures[correlation_id]
                
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """
        Get a workflow execution by ID.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            WorkflowExecution if found, None otherwise
        """
        return self.active_workflows.get(workflow_id)
        
    async def pause_workflow(self, workflow_id: str) -> bool:
        """
        Pause a workflow execution.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if paused successfully, False otherwise
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow or workflow.status not in [WorkflowExecutionStatus.RUNNING, WorkflowExecutionStatus.WAITING]:
            return False
            
        # Update status
        workflow.update_status(WorkflowExecutionStatus.PAUSED)
        
        # Create a checkpoint
        workflow.checkpoint()
        
        # Emit workflow paused event
        await self._emit_workflow_event(workflow_id, "workflow_paused", workflow)
        
        return True
        
    async def resume_workflow(self, workflow_id: str) -> bool:
        """
        Resume a paused workflow execution.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if resumed successfully, False otherwise
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow or workflow.status != WorkflowExecutionStatus.PAUSED:
            return False
            
        # Update status
        workflow.update_status(WorkflowExecutionStatus.RUNNING)
        
        # Emit workflow resumed event
        await self._emit_workflow_event(workflow_id, "workflow_resumed", workflow)
        
        # Get the workflow definition
        workflow_def = self.workflow_registry.get_workflow(workflow.workflow_type)
        if not workflow_def:
            workflow.set_error(f"Workflow type not found: {workflow.workflow_type}")
            return False
            
        # Continue workflow execution
        asyncio.create_task(self._continue_workflow(workflow, workflow_def))
        
        return True
        
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a workflow execution.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if canceled successfully, False otherwise
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow or workflow.status in [
            WorkflowExecutionStatus.COMPLETED,
            WorkflowExecutionStatus.FAILED,
            WorkflowExecutionStatus.CANCELED,
            WorkflowExecutionStatus.TIMED_OUT
        ]:
            return False
            
        # Update status
        workflow.update_status(WorkflowExecutionStatus.CANCELED)
        
        # Create a checkpoint
        workflow.checkpoint()
        
        # Emit workflow canceled event
        await self._emit_workflow_event(workflow_id, "workflow_canceled", workflow)
        
        # Complete the future if one exists
        if workflow_id in self.completion_futures:
            future = self.completion_futures[workflow_id]
            if not future.done():
                future.set_result(workflow)
                
        return True
        
    async def _checkpoint_workflows_periodically(self):
        """Periodically checkpoint all active workflows."""
        while True:
            try:
                # Create checkpoints for active workflows
                for workflow_id, workflow in self.active_workflows.items():
                    if workflow.status in [WorkflowExecutionStatus.RUNNING, WorkflowExecutionStatus.WAITING, WorkflowExecutionStatus.PAUSED]:
                        if workflow.should_checkpoint():
                            workflow.checkpoint()
                            
            except Exception as e:
                logger.error(f"Error in workflow checkpoint task: {str(e)}")
                
            # Wait before next checkpoint cycle
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _cleanup_workflows_periodically(self):
        """Periodically clean up completed workflows."""
        while True:
            try:
                # Find workflows to clean up
                now = datetime.datetime.now()
                workflows_to_remove = []
                
                for workflow_id, workflow in self.active_workflows.items():
                    if workflow.status in [
                        WorkflowExecutionStatus.COMPLETED,
                        WorkflowExecutionStatus.FAILED,
                        WorkflowExecutionStatus.CANCELED,
                        WorkflowExecutionStatus.TIMED_OUT
                    ]:
                        if workflow.completed_at:
                            # Calculate age in hours
                            completed_time = datetime.datetime.fromisoformat(workflow.completed_at)
                            age_hours = (now - completed_time).total_seconds() / 3600
                            
                            # Clean up workflows older than 24 hours
                            if age_hours > 24:
                                workflows_to_remove.append(workflow_id)
                                
                # Remove old workflows
                for workflow_id in workflows_to_remove:
                    logger.info(f"Cleaning up completed workflow {workflow_id}")
                    del self.active_workflows[workflow_id]
                    
                    # Clean up futures
                    if workflow_id in self.completion_futures:
                        del self.completion_futures[workflow_id]
                        
                # Update active workflows gauge
                self.metrics.gauge("workflow.executions.active").set(len(self.active_workflows))
                
            except Exception as e:
                logger.error(f"Error in workflow cleanup task: {str(e)}")
                
            # Wait before next cleanup cycle
            await asyncio.sleep(3600)  # Clean up once per hour
            
    def subscribe_to_workflow_events(self, workflow_id: str, callback: Callable):
        """
        Subscribe to events for a specific workflow.
        
        Args:
            workflow_id: Workflow ID
            callback: Async callback function taking (event_type, data)
        """
        if workflow_id not in self.event_subscribers:
            self.event_subscribers[workflow_id] = []
            
        self.event_subscribers[workflow_id].append(callback)
        
    def unsubscribe_from_workflow_events(self, workflow_id: str, callback: Callable):
        """
        Unsubscribe from events for a specific workflow.
        
        Args:
            workflow_id: Workflow ID
            callback: Callback function to unsubscribe
        """
        if workflow_id in self.event_subscribers:
            if callback in self.event_subscribers[workflow_id]:
                self.event_subscribers[workflow_id].remove(callback)
                
            if not self.event_subscribers[workflow_id]:
                del self.event_subscribers[workflow_id]
                
    async def _emit_workflow_event(self, workflow_id: str, event_type: str, data: Any):
        """
        Emit an event for a workflow.
        
        Args:
            workflow_id: Workflow ID
            event_type: Type of event
            data: Event data
        """
        # Check if there are any subscribers
        if workflow_id in self.event_subscribers:
            # Create event data
            event = {
                "workflow_id": workflow_id,
                "event_type": event_type,
                "timestamp": datetime.datetime.now().isoformat(),
                "data": data
            }
            
            # Notify subscribers
            for callback in self.event_subscribers[workflow_id]:
                try:
                    await callback(event_type, event)
                except Exception as e:
                    logger.error(f"Error in workflow event subscriber: {str(e)}")
                    
    def restore_workflow_from_checkpoint(self, workflow_id: str, checkpoint_index: int = -1) -> bool:
        """
        Restore a workflow from a checkpoint.
        
        Args:
            workflow_id: Workflow ID
            checkpoint_index: Index of the checkpoint to restore (-1 for latest)
            
        Returns:
            True if restored successfully, False otherwise
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False
            
        return workflow.restore_checkpoint(checkpoint_index)
        
    async def retry_workflow(self, workflow_id: str) -> bool:
        """
        Retry a failed workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if retry started successfully, False otherwise
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow or workflow.status != WorkflowExecutionStatus.FAILED:
            return False
            
        # Reset status
        workflow.update_status(WorkflowExecutionStatus.RUNNING)
        
        # Find last completed activity
        last_completed_index = -1
        for i, activity in enumerate(workflow.activities):
            if activity.get("status") == ActivityStatus.COMPLETED:
                last_completed_index = i
                
        # Set current activity index to last completed
        workflow.current_activity_index = last_completed_index
        
        # Get the workflow definition
        workflow_def = self.workflow_registry.get_workflow(workflow.workflow_type)
        if not workflow_def:
            workflow.set_error(f"Workflow type not found: {workflow.workflow_type}")
            return False
            
        # Continue workflow execution
        asyncio.create_task(self._continue_workflow(workflow, workflow_def))
        
        return True


# Create singleton instance
_workflow_engine_instance = None

def get_workflow_engine() -> WorkflowEngine:
    """Get or create the workflow engine singleton instance."""
    global _workflow_engine_instance
    if _workflow_engine_instance is None:
        _workflow_engine_instance = WorkflowEngine()
    return _workflow_engine_instance

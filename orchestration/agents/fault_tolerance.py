"""
Fault Tolerance Mechanisms for the Mathematical Multimodal LLM System.

This module provides fault tolerance capabilities to handle agent failures,
timeouts, and recovery mechanisms.
"""
import asyncio
import datetime
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import time
import json

from ..message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message, create_error_response
)
from ..message_bus.rabbitmq_wrapper import get_message_bus
from ..monitoring.logger import get_logger
from .registry import get_agent_registry
from .load_balancer import get_load_balancer

logger = get_logger(__name__)


class FaultToleranceManager:
    """
    Fault tolerance manager for handling agent failures.
    
    This component provides mechanisms for detecting and recovering from
    agent failures, including retries, timeouts, and failover.
    """
    
    def __init__(self):
        """Initialize the fault tolerance manager."""
        self.agent_registry = get_agent_registry()
        self.message_bus = get_message_bus()
        self.load_balancer = get_load_balancer()
        
        # Tracking for failed requests
        self.failed_requests: Dict[str, Dict[str, Any]] = {}
        
        # Retry policies (max_retries, backoff_factor, jitter)
        self.retry_policies: Dict[str, Tuple[int, float, float]] = {
            "default": (3, 1.5, 0.2),            # Default policy
            "critical": (5, 1.2, 0.1),           # More retries, less backoff for critical
            "computation": (2, 2.0, 0.3),        # Fewer retries, more backoff for computation
            "visualization": (2, 2.0, 0.3),      # Fewer retries, more backoff for visualization
            "ocr": (3, 1.5, 0.2),                # Standard for OCR
            "search": (3, 1.5, 0.2)              # Standard for search
        }
        
        # Health check settings
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5    # seconds
        
        # Recovery handlers
        self.recovery_handlers: Dict[str, List[Callable]] = {}
        
    async def with_retry(
        self,
        action: Callable,
        retry_policy: str = "default",
        context: Dict[str, Any] = None
    ) -> Tuple[bool, Any]:
        """
        Execute an action with retry logic.
        
        Args:
            action: Async callable that returns (success, result)
            retry_policy: Name of retry policy to use
            context: Context information for logging and recovery
            
        Returns:
            Tuple of (success, result)
        """
        # Get retry policy parameters
        policy = self.retry_policies.get(retry_policy, self.retry_policies["default"])
        max_retries, backoff_factor, jitter = policy
        
        # Initialize attempt counter
        attempt = 0
        
        # Keep track of failed attempts
        failed_attempts = []
        
        # Loop until success or max retries reached
        while attempt <= max_retries:
            attempt += 1
            
            try:
                # Execute the action
                success, result = await action()
                
                # If successful, return immediately
                if success:
                    return True, result
                    
                # Action failed but didn't raise an exception
                error_info = {
                    "attempt": attempt,
                    "error": "Action returned failure",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "context": context
                }
                failed_attempts.append(error_info)
                
                # Log the failure
                logger.warning(
                    f"Attempt {attempt}/{max_retries+1} failed: Action returned failure. "
                    f"Context: {json.dumps(context) if context else 'None'}"
                )
                
            except Exception as e:
                # Action raised an exception
                error_info = {
                    "attempt": attempt,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "context": context
                }
                failed_attempts.append(error_info)
                
                # Log the exception
                logger.warning(
                    f"Attempt {attempt}/{max_retries+1} failed with exception: {str(e)}. "
                    f"Context: {json.dumps(context) if context else 'None'}"
                )
                
            # Check if we've reached max retries
            if attempt > max_retries:
                # No more retries, return failure
                logger.error(
                    f"Action failed after {attempt} attempts. "
                    f"Context: {json.dumps(context) if context else 'None'}"
                )
                
                # Record failure with correlation ID if available
                correlation_id = context.get("correlation_id") if context else None
                if correlation_id:
                    self.failed_requests[correlation_id] = {
                        "attempts": failed_attempts,
                        "max_retries": max_retries,
                        "retry_policy": retry_policy,
                        "context": context,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                # Return the last result if available
                if failed_attempts and "result" in locals():
                    return False, result
                else:
                    return False, None
                    
            # Calculate backoff time with jitter
            backoff_time = backoff_factor ** (attempt - 1)
            jitter_amount = backoff_time * jitter
            backoff_time = backoff_time + random.uniform(-jitter_amount, jitter_amount)
            
            # Log the backoff
            logger.info(f"Retrying in {backoff_time:.2f} seconds...")
            
            # Wait before retry
            await asyncio.sleep(backoff_time)
            
    async def make_resilient_request(
        self,
        capability: str,
        request_type: MessageType,
        body: Dict[str, Any],
        retry_policy: str = "default",
        timeout: float = 30.0,
        context: Dict[str, Any] = None
    ) -> Tuple[bool, Optional[Message]]:
        """
        Make a request to an agent with fault tolerance.
        
        Args:
            capability: Required capability
            request_type: Type of request message
            body: Body of the request message
            retry_policy: Name of retry policy to use
            timeout: Timeout for each attempt in seconds
            context: Additional context information
            
        Returns:
            Tuple of (success, response_message)
        """
        # Combine context with request information
        request_context = {
            "capability": capability,
            "request_type": str(request_type),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if context:
            request_context.update(context)
            
        # List to track failed agents
        failed_agents = []
        
        # Define the action for retry logic
        async def make_request_action():
            # Select an agent for the capability, excluding failed ones
            agent_id = self.load_balancer.select_agent(
                capability=capability,
                consider_load=True,
                exclude_agents=failed_agents
            )
            
            if not agent_id:
                # No suitable agent found
                logger.error(f"No available agent with capability: {capability}")
                return False, None
                
            # Update context with selected agent
            request_context["agent_id"] = agent_id
            
            # Create request message
            message = create_message(
                message_type=request_type,
                sender="fault_tolerance_manager",
                recipient=agent_id,
                body=body.copy(),
                correlation_id=request_context["correlation_id"],
                priority=MessagePriority.NORMAL
            )
            
            # Send the request
            sent = await self.message_bus.send_message(message)
            if not sent:
                # Failed to send message
                logger.error(f"Failed to send request to agent: {agent_id}")
                failed_agents.append(agent_id)
                return False, None
                
            # Wait for response
            try:
                response_future = asyncio.Future()
                self.message_bus.response_handlers[request_context["correlation_id"]] = response_future
                
                response = await asyncio.wait_for(response_future, timeout=timeout)
                
                # Clean up
                if request_context["correlation_id"] in self.message_bus.response_handlers:
                    del self.message_bus.response_handlers[request_context["correlation_id"]]
                    
                # Check if it's an error response
                if response.header.message_type == MessageType.ERROR:
                    logger.warning(f"Received error response from agent {agent_id}: {response.body.get('error_message', 'Unknown error')}")
                    failed_agents.append(agent_id)
                    return False, response
                    
                # Success
                return True, response
                
            except asyncio.TimeoutError:
                # Timeout waiting for response
                logger.warning(f"Request to agent {agent_id} timed out after {timeout} seconds")
                failed_agents.append(agent_id)
                
                # Clean up
                if request_context["correlation_id"] in self.message_bus.response_handlers:
                    del self.message_bus.response_handlers[request_context["correlation_id"]]
                    
                return False, None
                
            except Exception as e:
                # Other error
                logger.error(f"Error waiting for response from agent {agent_id}: {str(e)}")
                failed_agents.append(agent_id)
                
                # Clean up
                if request_context["correlation_id"] in self.message_bus.response_handlers:
                    del self.message_bus.response_handlers[request_context["correlation_id"]]
                    
                return False, None
                
        # Use the retry logic
        return await self.with_retry(
            action=make_request_action,
            retry_policy=retry_policy,
            context=request_context
        )
        
    async def health_check(self, agent_id: str) -> bool:
        """
        Perform a health check on an agent.
        
        Args:
            agent_id: Agent ID to check
            
        Returns:
            True if agent is healthy, False otherwise
        """
        # Create a health check message
        message = create_message(
            message_type=MessageType.HEARTBEAT,
            sender="fault_tolerance_manager",
            recipient=agent_id,
            body={"health_check": True},
            correlation_id=str(uuid.uuid4()),
            priority=MessagePriority.HIGH
        )
        
        # Send the health check
        sent = await self.message_bus.send_message(message)
        if not sent:
            logger.warning(f"Failed to send health check to agent: {agent_id}")
            return False
            
        # Wait for response
        try:
            response_future = asyncio.Future()
            self.message_bus.response_handlers[message.header.correlation_id] = response_future
            
            await asyncio.wait_for(response_future, timeout=self.health_check_timeout)
            
            # Clean up
            del self.message_bus.response_handlers[message.header.correlation_id]
            
            # Agent is healthy
            return True
            
        except asyncio.TimeoutError:
            # Timeout waiting for response
            logger.warning(f"Health check to agent {agent_id} timed out")
            
            # Clean up
            if message.header.correlation_id in self.message_bus.response_handlers:
                del self.message_bus.response_handlers[message.header.correlation_id]
                
            return False
            
        except Exception as e:
            # Other error
            logger.error(f"Error in health check for agent {agent_id}: {str(e)}")
            
            # Clean up
            if message.header.correlation_id in self.message_bus.response_handlers:
                del self.message_bus.response_handlers[message.header.correlation_id]
                
            return False
            
    async def monitor_agents(self):
        """
        Periodically monitor agent health.
        
        This method runs in a background task and checks agent health
        at regular intervals.
        """
        while True:
            try:
                # Get all active agents
                active_agents = self.agent_registry.get_active_agents()
                
                for agent in active_agents:
                    agent_id = agent["agent_id"]
                    
                    # Check agent health
                    healthy = await self.health_check(agent_id)
                    
                    if not healthy:
                        # Agent is unhealthy
                        logger.warning(f"Agent {agent_id} is unhealthy")
                        
                        # Mark as inactive in registry
                        self.agent_registry.update_agent_status(agent_id, "inactive")
                        
                        # Trigger recovery for this agent
                        await self._trigger_agent_recovery(agent_id, agent)
                        
            except Exception as e:
                logger.error(f"Error in agent monitoring: {str(e)}")
                
            # Wait before next check
            await asyncio.sleep(self.health_check_interval)
            
    async def _trigger_agent_recovery(self, agent_id: str, agent_info: Dict[str, Any]):
        """
        Trigger recovery mechanisms for an unhealthy agent.
        
        Args:
            agent_id: ID of the unhealthy agent
            agent_info: Agent information
        """
        agent_type = agent_info.get("agent_type", "unknown")
        
        # Call registered recovery handlers
        if agent_type in self.recovery_handlers:
            for handler in self.recovery_handlers[agent_type]:
                try:
                    await handler(agent_id, agent_info)
                except Exception as e:
                    logger.error(f"Error in recovery handler for agent {agent_id}: {str(e)}")
                    
        # Call generic recovery handlers
        if "all" in self.recovery_handlers:
            for handler in self.recovery_handlers["all"]:
                try:
                    await handler(agent_id, agent_info)
                except Exception as e:
                    logger.error(f"Error in recovery handler for agent {agent_id}: {str(e)}")
                    
    def register_recovery_handler(self, agent_type: str, handler: Callable):
        """
        Register a recovery handler for an agent type.
        
        Args:
            agent_type: Agent type or 'all' for all types
            handler: Async function that takes (agent_id, agent_info)
        """
        if agent_type not in self.recovery_handlers:
            self.recovery_handlers[agent_type] = []
        self.recovery_handlers[agent_type].append(handler)
        
    def set_retry_policy(self, name: str, max_retries: int, backoff_factor: float, jitter: float):
        """
        Set a custom retry policy.
        
        Args:
            name: Policy name
            max_retries: Maximum number of retries
            backoff_factor: Base for exponential backoff
            jitter: Jitter factor (0.0 to 1.0)
        """
        self.retry_policies[name] = (max_retries, backoff_factor, jitter)
        
    def get_retry_policy(self, name: str) -> Tuple[int, float, float]:
        """
        Get a retry policy.
        
        Args:
            name: Policy name
            
        Returns:
            Tuple of (max_retries, backoff_factor, jitter)
        """
        return self.retry_policies.get(name, self.retry_policies["default"])


# Create singleton instance
_fault_tolerance_manager = None

def get_fault_tolerance_manager() -> FaultToleranceManager:
    """Get the fault tolerance manager singleton."""
    global _fault_tolerance_manager
    if _fault_tolerance_manager is None:
        _fault_tolerance_manager = FaultToleranceManager()
    return _fault_tolerance_manager


# Missing import for random module
import random

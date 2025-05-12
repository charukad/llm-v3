"""
Base Agent Implementation for the Mathematical Multimodal LLM System.

This module provides a base agent class that all specialized agents can inherit
from, providing common functionality for messaging, capabilities, and integration.
"""
import asyncio
import os
import sys
import logging
import json
import datetime
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import traceback

from ..message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message, create_error_response
)
from ..message_bus.rabbitmq_wrapper import get_message_bus
from ..monitoring.logger import get_logger, set_correlation_id
from ..monitoring.tracing import get_tracer, Span
from ..monitoring.metrics import get_registry
from .communication import create_agent_communication
from .registry import get_agent_registry

logger = get_logger(__name__)


class BaseAgent:
    """
    Base agent implementation with common functionality.
    
    This class provides common infrastructure for all agent types,
    including message handling, capability advertisement, and integration.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        description: str = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (llm, computation, ocr, etc.)
            capabilities: List of capabilities the agent provides
            description: Optional agent description
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.description = description or f"{agent_type.capitalize()} agent"
        
        # Infrastructure components
        self.message_bus = get_message_bus()
        self.agent_registry = get_agent_registry()
        self.communication = create_agent_communication(agent_id)
        self.tracer = get_tracer()
        self.metrics = get_registry()
        
        # State tracking
        self.running = False
        self.load = 0.0
        self.status = "initializing"
        self.last_heartbeat = None
        
        # Message handlers for different message types
        self.message_handlers: Dict[str, Callable] = {}
        
        # Register standard message handlers
        self._register_standard_handlers()
        
    def _register_standard_handlers(self):
        """Register standard message handlers for common message types."""
        # Heartbeat handler
        self.register_message_handler(
            MessageType.HEARTBEAT,
            self._handle_heartbeat
        )
        
        # Status update handler
        self.register_message_handler(
            MessageType.STATUS_UPDATE,
            self._handle_status_update
        )
        
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Async function that takes a Message and returns None
        """
        self.message_handlers[str(message_type)] = handler
        
    async def start(self):
        """Start the agent."""
        logger.info(f"Starting agent {self.agent_id} ({self.agent_type})")
        
        try:
            # Connect to message bus if not already connected
            if not self.message_bus.connection:
                await self.message_bus.connect()
                
            # Setup agent queue
            queue = await self.message_bus.setup_agent_queues(self.agent_id, self.capabilities)
            
            # Setup consumer
            await self.message_bus.setup_consumer(
                f"agent.{self.agent_id}",
                self._handle_message
            )
            
            # Subscribe to broadcast messages
            broadcast_queue = await self.message_bus.declare_queue(
                f"broadcast.{self.agent_id}",
                durable=False,
                exclusive=True
            )
            
            await self.message_bus.bind_queue(
                f"broadcast.{self.agent_id}",
                "broadcast.#"
            )
            
            await self.message_bus.setup_consumer(
                f"broadcast.{self.agent_id}",
                self._handle_message
            )
            
            # Subscribe to capability-based routing
            for capability in self.capabilities:
                capability_queue = await self.message_bus.declare_queue(
                    f"capability.{capability}.{self.agent_id}",
                    durable=False,
                    exclusive=True
                )
                
                await self.message_bus.bind_queue(
                    f"capability.{capability}.{self.agent_id}",
                    f"capability.{capability}"
                )
                
                await self.message_bus.setup_consumer(
                    f"capability.{capability}.{self.agent_id}",
                    self._handle_message
                )
                
            # Advertise capabilities
            await self.advertise_capabilities()
            
            # Start heartbeat task
            self.running = True
            self.status = "active"
            asyncio.create_task(self._heartbeat_task())
            
            # Initialize agent-specific components
            await self.initialize()
            
            logger.info(f"Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting agent {self.agent_id}: {str(e)}")
            traceback.print_exc()
            self.status = "error"
            return False
            
    async def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")
        
        # Stop the heartbeat task
        self.running = False
        
        # Update status
        self.status = "stopped"
        try:
            # Advertise stopped status
            await self.communication.update_status(
                status="stopped",
                load=0.0,
                metadata={"shutdown_time": datetime.datetime.now().isoformat()}
            )
            
            # Perform agent-specific cleanup
            await self.cleanup()
            
        except Exception as e:
            logger.error(f"Error stopping agent {self.agent_id}: {str(e)}")
            
        logger.info(f"Agent {self.agent_id} stopped")
        
    async def initialize(self):
        """
        Initialize agent-specific components.
        
        This method should be overridden by specific agent implementations.
        """
        pass
        
    async def cleanup(self):
        """
        Perform agent-specific cleanup.
        
        This method should be overridden by specific agent implementations.
        """
        pass
        
    async def _handle_message(self, message: Message):
        """
        Handle incoming messages.
        
        This is the main message processor that routes messages to
        appropriate handlers based on message type.
        
        Args:
            message: The received message
        """
        # Set correlation ID for logging context
        set_correlation_id(message.header.correlation_id)
        
        # Extract message info for metrics
        sender = message.header.route.sender
        message_type = str(message.header.message_type)
        
        # Log message receipt
        logger.debug(f"Agent {self.agent_id} received {message_type} message from {sender}")
        
        # Track message processing
        with self.tracer.span(
            f"agent_message_processing.{message_type}",
            metadata={
                "agent_id": self.agent_id,
                "message_type": message_type,
                "sender": sender
            }
        ) as span:
            try:
                # Record message receipt in metrics
                self.metrics.counter(
                    "agent.messages.received",
                    labels={"agent_id": self.agent_id, "message_type": message_type}
                ).increment()
                
                # First, pass to communication module to handle responses and subscriptions
                await self.communication.handle_message(message)
                
                # Check if we have a specific handler for this message type
                if message_type in self.message_handlers:
                    # Call the registered handler
                    await self.message_handlers[message_type](message)
                else:
                    # Use the default handler
                    await self.handle_message(message)
                    
                # Record successful processing
                self.metrics.counter(
                    "agent.messages.processed",
                    labels={"agent_id": self.agent_id, "message_type": message_type}
                ).increment()
                
            except Exception as e:
                # Log the error
                logger.error(f"Error processing {message_type} message: {str(e)}")
                traceback.print_exc()
                
                # Record error in metrics
                self.metrics.counter(
                    "agent.messages.errors",
                    labels={"agent_id": self.agent_id, "message_type": message_type}
                ).increment()
                
                # Send error response if appropriate
                if message.header.route.reply_to:
                    try:
                        error_response = create_error_response(
                            sender=self.agent_id,
                            recipient=message.header.route.reply_to,
                            error_message=f"Error processing message: {str(e)}",
                            error_code="PROCESSING_ERROR",
                            original_message_id=message.header.message_id
                        )
                        
                        await self.message_bus.send_message(error_response)
                        
                    except Exception as e2:
                        logger.error(f"Error sending error response: {str(e2)}")
                        
            finally:
                # Clear correlation ID
                set_correlation_id(None)
                
    async def handle_message(self, message: Message):
        """
        Handle a message with no specific handler.
        
        This method should be overridden by specific agent implementations.
        
        Args:
            message: The received message
        """
        # Default implementation just logs unhandled message
        logger.warning(
            f"Agent {self.agent_id} has no handler for message type {message.header.message_type}"
        )
        
        # If the message expects a response, send an error
        if message.header.route.reply_to:
            error_response = create_error_response(
                sender=self.agent_id,
                recipient=message.header.route.reply_to,
                error_message=f"Unsupported message type: {message.header.message_type}",
                error_code="UNSUPPORTED_MESSAGE_TYPE",
                original_message_id=message.header.message_id
            )
            
            await self.message_bus.send_message(error_response)
            
    async def _handle_heartbeat(self, message: Message):
        """
        Handle heartbeat messages.
        
        Args:
            message: Heartbeat message
        """
        # Check if it's a health check
        if message.body.get("health_check", False):
            # Send a response with current status
            response = create_message(
                message_type=MessageType.HEARTBEAT,
                sender=self.agent_id,
                recipient=message.header.route.sender,
                body={
                    "status": self.status,
                    "load": self.load,
                    "timestamp": datetime.datetime.now().isoformat()
                },
                correlation_id=message.header.correlation_id
            )
            
            await self.message_bus.send_message(response)
            
    async def _handle_status_update(self, message: Message):
        """
        Handle status update messages.
        
        Args:
            message: Status update message
        """
        # Default implementation does nothing with status updates
        pass
        
    async def _heartbeat_task(self):
        """Periodically send heartbeat messages."""
        while self.running:
            try:
                # Update load (should be overridden by specific agents)
                self.load = await self.calculate_load()
                
                # Record the load
                self.metrics.gauge(
                    "agent.load",
                    labels={"agent_id": self.agent_id, "agent_type": self.agent_type}
                ).set(self.load)
                
                # Send heartbeat
                await self.communication.send_heartbeat(self.load)
                
                # Update last heartbeat time
                self.last_heartbeat = datetime.datetime.now()
                
            except Exception as e:
                logger.error(f"Error in heartbeat task: {str(e)}")
                
            # Wait before next heartbeat
            await asyncio.sleep(15)  # 15 seconds between heartbeats
            
    async def calculate_load(self) -> float:
        """
        Calculate the current load of the agent.
        
        This method should be overridden by specific agent implementations
        to provide more accurate load calculations.
        
        Returns:
            Load value between 0.0 and 1.0
        """
        # Default implementation returns a fixed value
        return 0.5
        
    async def advertise_capabilities(self) -> bool:
        """
        Advertise agent capabilities to the system.
        
        Returns:
            True if successful, False otherwise
        """
        metadata = {
            "agent_type": self.agent_type,
            "description": self.description,
            "version": "1.0",
            "start_time": datetime.datetime.now().isoformat()
        }
        
        return await self.communication.advertise_capabilities(
            self.capabilities,
            metadata
        )
        
    async def update_status(self, status: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Update agent status in the system.
        
        Args:
            status: New status value
            metadata: Additional status metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Update local status
        self.status = status
        
        # Broadcast status update
        return await self.communication.update_status(
            status=status,
            load=self.load,
            metadata=metadata
        )
        
    async def send_broadcast(
        self,
        message_type: MessageType,
        body: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Send a broadcast message.
        
        Args:
            message_type: Type of message to broadcast
            body: Message body
            priority: Message priority
            
        Returns:
            True if successful, False otherwise
        """
        return await self.communication.broadcast(
            message_type=message_type,
            body=body,
            priority=priority
        )
        
    async def send_request(
        self,
        recipient: str,
        request_type: MessageType,
        body: Dict[str, Any],
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Tuple[bool, Optional[Message]]:
        """
        Send a request to another agent.
        
        Args:
            recipient: Recipient agent ID
            request_type: Type of request
            body: Request body
            timeout: Timeout in seconds
            priority: Message priority
            
        Returns:
            Tuple of (success, response)
        """
        return await self.communication.request(
            recipient=recipient,
            request_type=request_type,
            body=body,
            timeout=timeout,
            priority=priority
        )
        
    async def send_computation_request(
        self,
        expression: str,
        operation: str,
        variables: List[str] = None,
        domain: str = None,
        step_by_step: bool = False,
        timeout: float = 60.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Send a computation request to a math agent.
        
        Args:
            expression: Mathematical expression
            operation: Operation to perform
            variables: Variables in the expression
            domain: Mathematical domain
            step_by_step: Whether to generate step-by-step solution
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, result)
        """
        body = {
            "expression": expression,
            "operation": operation,
            "step_by_step": step_by_step
        }
        
        if variables:
            body["variables"] = variables
            
        if domain:
            body["domain"] = domain
            
        # First, find a suitable math agent
        math_agents = self.agent_registry.find_agents_by_capability("compute")
        if not math_agents:
            logger.error("No math computation agents available")
            return False, {"error": "No computation agent available"}
            
        # Send the request
        success, response = await self.send_request(
            recipient=math_agents[0],
            request_type=MessageType.COMPUTATION_REQUEST,
            body=body,
            timeout=timeout
        )
        
        if success and response:
            return True, response.body
        else:
            return False, {"error": "Computation request failed"}
            
    async def send_visualization_request(
        self,
        visualization_type: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any] = None,
        timeout: float = 60.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Send a visualization request to a visualization agent.
        
        Args:
            visualization_type: Type of visualization
            data: Data to visualize
            parameters: Visualization parameters
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, result)
        """
        body = {
            "visualization_type": visualization_type,
            "data": data,
            "parameters": parameters or {}
        }
        
        # Find a suitable visualization agent
        viz_agents = self.agent_registry.find_agents_by_capability("generate_visualization")
        if not viz_agents:
            logger.error("No visualization agents available")
            return False, {"error": "No visualization agent available"}
            
        # Send the request
        success, response = await self.send_request(
            recipient=viz_agents[0],
            request_type=MessageType.VISUALIZATION_REQUEST,
            body=body,
            timeout=timeout
        )
        
        if success and response:
            return True, response.body
        else:
            return False, {"error": "Visualization request failed"}

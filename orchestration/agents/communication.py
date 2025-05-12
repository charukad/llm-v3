"""
Agent-to-Agent Communication Patterns for the Mathematical Multimodal LLM System.

This module provides standardized communication patterns for direct interaction
between agents, including request-response, broadcasts, and subscriptions.
"""
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, TypeVar, Tuple
import logging
import datetime
import json

from ..message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message, create_error_response
)
from ..message_bus.rabbitmq_wrapper import get_message_bus
from ..monitoring.logger import get_logger
from ..monitoring.tracing import get_tracer, Span
from ..monitoring.metrics import get_registry, record_processing_time

logger = get_logger(__name__)


class AgentCommunication:
    """
    Agent communication utility for standardized interaction patterns.
    
    This class provides high-level communication methods that implement
    common interaction patterns between agents, such as request-response,
    broadcast notifications, and subscriptions.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize the agent communication utility.
        
        Args:
            agent_id: The ID of the agent using this communication utility
        """
        self.agent_id = agent_id
        self.message_bus = get_message_bus()
        self.tracer = get_tracer()
        self.metrics = get_registry()
        
        # Pending requests and their futures
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Subscription handlers
        self.subscription_handlers: Dict[str, List[Callable]] = {}
        
    async def request(
        self,
        recipient: str,
        request_type: MessageType,
        body: Dict[str, Any],
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
        conversation_id: Optional[str] = None,
        flow_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[Message]]:
        """
        Send a request to another agent and wait for a response.
        
        Args:
            recipient: The ID of the recipient agent
            request_type: The type of request message
            body: The body of the request message
            timeout: The timeout in seconds for waiting for a response
            priority: The priority of the request
            conversation_id: Optional conversation ID for tracking
            flow_id: Optional workflow ID for tracking
            
        Returns:
            Tuple of (success, response_message)
        """
        correlation_id = str(uuid.uuid4())
        
        with self.tracer.span(
            f"agent_request.{request_type}",
            metadata={
                "agent_id": self.agent_id,
                "recipient": recipient,
                "request_type": str(request_type)
            }
        ) as span:
            # Create request message
            request_message = create_message(
                message_type=request_type,
                sender=self.agent_id,
                recipient=recipient,
                body=body,
                priority=priority,
                correlation_id=correlation_id,
                conversation_id=conversation_id,
                flow_id=flow_id,
                reply_to=self.agent_id
            )
            
            # Create a future for the response
            response_future = asyncio.Future()
            self.pending_requests[correlation_id] = response_future
            
            # Record the request in metrics
            self.metrics.counter(
                "agent.requests.sent",
                labels={"agent_id": self.agent_id, "request_type": str(request_type)}
            ).increment()
            
            # Send the request
            start_time = datetime.datetime.now()
            sent = await self.message_bus.send_message(request_message)
            
            if not sent:
                span.add_metadata("error", "Failed to send request message")
                del self.pending_requests[correlation_id]
                logger.error(f"Failed to send request to {recipient}")
                
                # Record failure in metrics
                self.metrics.counter(
                    "agent.requests.failed",
                    labels={"agent_id": self.agent_id, "reason": "send_failure"}
                ).increment()
                
                return False, None
                
            # Wait for the response
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                
                # Calculate and record latency
                end_time = datetime.datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                self.metrics.histogram(
                    "agent.request.latency",
                    labels={"agent_id": self.agent_id, "request_type": str(request_type)}
                ).observe(latency_ms)
                
                span.add_metadata("latency_ms", latency_ms)
                span.add_metadata("response_type", str(response.header.message_type))
                
                # Record success in metrics
                self.metrics.counter(
                    "agent.requests.succeeded",
                    labels={"agent_id": self.agent_id, "request_type": str(request_type)}
                ).increment()
                
                return True, response
                
            except asyncio.TimeoutError:
                span.add_metadata("error", "Request timed out")
                del self.pending_requests[correlation_id]
                logger.warning(f"Request to {recipient} timed out after {timeout} seconds")
                
                # Record timeout in metrics
                self.metrics.counter(
                    "agent.requests.failed",
                    labels={"agent_id": self.agent_id, "reason": "timeout"}
                ).increment()
                
                return False, None
                
            except Exception as e:
                span.add_metadata("error", str(e))
                if correlation_id in self.pending_requests:
                    del self.pending_requests[correlation_id]
                logger.error(f"Error waiting for response from {recipient}: {str(e)}")
                
                # Record error in metrics
                self.metrics.counter(
                    "agent.requests.failed",
                    labels={"agent_id": self.agent_id, "reason": "error"}
                ).increment()
                
                return False, None
                
    async def broadcast(
        self,
        message_type: MessageType,
        body: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        conversation_id: Optional[str] = None,
        flow_id: Optional[str] = None,
    ) -> bool:
        """
        Broadcast a message to all listening agents.
        
        Args:
            message_type: The type of message to broadcast
            body: The body of the message
            priority: The priority of the message
            conversation_id: Optional conversation ID for tracking
            flow_id: Optional workflow ID for tracking
            
        Returns:
            True if the broadcast was successful, False otherwise
        """
        with self.tracer.span(
            f"agent_broadcast.{message_type}",
            metadata={
                "agent_id": self.agent_id,
                "message_type": str(message_type)
            }
        ) as span:
            # Create broadcast message
            broadcast_message = create_message(
                message_type=message_type,
                sender=self.agent_id,
                recipient="broadcast",
                body=body,
                priority=priority,
                conversation_id=conversation_id,
                flow_id=flow_id,
                broadcast=True
            )
            
            # Record the broadcast in metrics
            self.metrics.counter(
                "agent.broadcasts.sent",
                labels={"agent_id": self.agent_id, "message_type": str(message_type)}
            ).increment()
            
            # Send the broadcast
            sent = await self.message_bus.send_message(broadcast_message)
            
            if not sent:
                span.add_metadata("error", "Failed to send broadcast message")
                logger.error("Failed to send broadcast message")
                
                # Record failure in metrics
                self.metrics.counter(
                    "agent.broadcasts.failed",
                    labels={"agent_id": self.agent_id}
                ).increment()
                
            return sent
            
    async def subscribe(
        self,
        message_type: MessageType,
        handler: Callable[[Message], None]
    ):
        """
        Subscribe to messages of a specific type.
        
        Args:
            message_type: The type of message to subscribe to
            handler: The handler function to call when a message is received
        """
        # Register handler
        if str(message_type) not in self.subscription_handlers:
            self.subscription_handlers[str(message_type)] = []
        self.subscription_handlers[str(message_type)].append(handler)
        
        # Register with message bus for this type if not already done
        # This is a local tracking, message bus has its own subscription mechanism
        logger.info(f"Agent {self.agent_id} subscribed to message type {message_type}")
        
    async def unsubscribe(
        self,
        message_type: MessageType,
        handler: Optional[Callable[[Message], None]] = None
    ):
        """
        Unsubscribe from messages of a specific type.
        
        Args:
            message_type: The type of message to unsubscribe from
            handler: The specific handler to remove, or None to remove all
        """
        msg_type_str = str(message_type)
        if msg_type_str not in self.subscription_handlers:
            return
            
        if handler is None:
            # Remove all handlers
            del self.subscription_handlers[msg_type_str]
        else:
            # Remove specific handler
            if handler in self.subscription_handlers[msg_type_str]:
                self.subscription_handlers[msg_type_str].remove(handler)
                
            # Clean up empty handler list
            if not self.subscription_handlers[msg_type_str]:
                del self.subscription_handlers[msg_type_str]
                
        logger.info(f"Agent {self.agent_id} unsubscribed from message type {message_type}")
        
    async def handle_message(self, message: Message):
        """
        Handle incoming messages.
        
        This method should be called by the agent when a message is received.
        It processes the message based on its type and dispatches it to
        appropriate handlers.
        
        Args:
            message: The received message
        """
        # Handle response messages
        if message.header.correlation_id in self.pending_requests:
            future = self.pending_requests[message.header.correlation_id]
            if not future.done():
                future.set_result(message)
            del self.pending_requests[message.header.correlation_id]
            return
            
        # Handle subscription messages
        message_type = str(message.header.message_type)
        if message_type in self.subscription_handlers:
            for handler in self.subscription_handlers[message_type]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in subscription handler for {message_type}: {str(e)}")
                    
    async def advertise_capabilities(
        self,
        capabilities: List[str],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Advertise agent capabilities to the system.
        
        Args:
            capabilities: List of capabilities the agent supports
            metadata: Additional metadata about the agent
            
        Returns:
            True if the advertisement was successful, False otherwise
        """
        body = {
            "agent_id": self.agent_id,
            "capabilities": capabilities,
            "status": "active",
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return await self.broadcast(
            message_type=MessageType.CAPABILITY_ADVERTISEMENT,
            body=body,
            priority=MessagePriority.NORMAL
        )
        
    async def update_status(
        self,
        status: str,
        load: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Update agent status in the system.
        
        Args:
            status: Current agent status (active, busy, error, etc.)
            load: Current agent load (0.0 to 1.0)
            metadata: Additional status metadata
            
        Returns:
            True if the update was successful, False otherwise
        """
        body = {
            "agent_id": self.agent_id,
            "status": status,
            "load": load,
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return await self.broadcast(
            message_type=MessageType.STATUS_UPDATE,
            body=body,
            priority=MessagePriority.LOW
        )
        
    async def send_heartbeat(self, load: float = 0.0) -> bool:
        """
        Send a heartbeat message to indicate the agent is alive.
        
        Args:
            load: Current agent load (0.0 to 1.0)
            
        Returns:
            True if the heartbeat was successful, False otherwise
        """
        body = {
            "agent_id": self.agent_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "load": load
        }
        
        return await self.broadcast(
            message_type=MessageType.HEARTBEAT,
            body=body,
            priority=MessagePriority.LOW
        )


# Factory function to create agent communication instances
def create_agent_communication(agent_id: str) -> AgentCommunication:
    """
    Create an agent communication utility for the specified agent.
    
    Args:
        agent_id: The ID of the agent
        
    Returns:
        AgentCommunication instance
    """
    return AgentCommunication(agent_id)

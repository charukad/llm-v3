"""
Message handler for the Multi-agent Communication Protocol (MCP).

This module provides functionality for routing, processing, and tracking messages
between agents in the system.
"""
from typing import Dict, Any, Optional, List, Callable, Set
import logging
import time
import datetime
import asyncio
from .message_formats import Message, MessageType, MessagePriority, create_error_response
from ..monitoring.logger import get_logger

logger = get_logger(__name__)


class MessageRouter:
    """
    Routes messages between agents based on content, capabilities, and load.
    """
    def __init__(self):
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.route_handlers: Dict[str, List[Callable]] = {}
        self.broadcast_handlers: Dict[MessageType, List[Callable]] = {}
        
    def register_agent(self, agent_id: str, capabilities: List[str], metadata: Dict[str, Any] = None):
        """Register an agent and its capabilities."""
        self.agent_capabilities[agent_id] = set(capabilities)
        self.agent_status[agent_id] = {
            "status": "active",
            "last_heartbeat": datetime.datetime.now().isoformat(),
            "load": 0.0,
            "metadata": metadata or {},
            "registered_at": datetime.datetime.now().isoformat()
        }
        logger.info(f"Agent {agent_id} registered with capabilities: {capabilities}")
    
    def deregister_agent(self, agent_id: str):
        """Deregister an agent."""
        if agent_id in self.agent_capabilities:
            del self.agent_capabilities[agent_id]
        if agent_id in self.agent_status:
            del self.agent_status[agent_id]
        logger.info(f"Agent {agent_id} deregistered")
    
    def update_agent_status(self, agent_id: str, status: str, load: float = None, metadata: Dict[str, Any] = None):
        """Update the status of an agent."""
        if agent_id not in self.agent_status:
            return
        
        self.agent_status[agent_id]["status"] = status
        self.agent_status[agent_id]["last_heartbeat"] = datetime.datetime.now().isoformat()
        
        if load is not None:
            self.agent_status[agent_id]["load"] = load
            
        if metadata:
            self.agent_status[agent_id]["metadata"].update(metadata)
    
    def find_agent_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability."""
        matching_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if capability in capabilities and self.agent_status[agent_id]["status"] == "active":
                matching_agents.append(agent_id)
        return matching_agents
    
    def get_optimal_agent(self, capability: str) -> Optional[str]:
        """Get the optimal agent for a capability based on load and status."""
        matching_agents = self.find_agent_by_capability(capability)
        if not matching_agents:
            return None
            
        # Sort by load (lower is better)
        matching_agents.sort(key=lambda agent_id: self.agent_status[agent_id]["load"])
        return matching_agents[0] if matching_agents else None
    
    def register_route_handler(self, route_key: str, handler: Callable):
        """Register a handler for a specific route."""
        if route_key not in self.route_handlers:
            self.route_handlers[route_key] = []
        self.route_handlers[route_key].append(handler)
    
    def register_broadcast_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for broadcast messages of a specific type."""
        if message_type not in self.broadcast_handlers:
            self.broadcast_handlers[message_type] = []
        self.broadcast_handlers[message_type].append(handler)
    
    def get_route_key(self, message: Message) -> str:
        """Get the routing key for a message."""
        return f"{message.header.route.sender}.{message.header.message_type}.{message.header.route.recipient}"
    
    def route_message(self, message: Message) -> bool:
        """
        Route a message to the appropriate handlers.
        
        Returns True if the message was successfully routed, False otherwise.
        """
        # Update message trace information
        message.header.trace.agent_hops.append({
            "agent": "message_router",
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "route"
        })
        
        # Increment hop count and check TTL
        message.header.route.hop_count += 1
        if message.header.route.hop_count > message.header.route.max_hops:
            logger.warning(f"Message {message.header.message_id} exceeded max hops: {message.header.route.hop_count}")
            return False
            
        # Handle broadcast messages
        if message.header.route.broadcast:
            message_type = message.header.message_type
            if message_type in self.broadcast_handlers:
                for handler in self.broadcast_handlers[message_type]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Error in broadcast handler for {message_type}: {str(e)}")
            return True
            
        # Route to specific handlers
        route_key = self.get_route_key(message)
        if route_key in self.route_handlers:
            for handler in self.route_handlers[route_key]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error in route handler for {route_key}: {str(e)}")
            return True
        
        # If no specific handlers, try finding by capability
        recipient = message.header.route.recipient
        if recipient in self.agent_capabilities:
            # Recipient exists, but no handler registered
            logger.warning(f"No handler registered for route {route_key}")
            return False
        
        # Try to find an alternate agent with required capability
        # This would require extracting capability from message content
        # For now, just log that the recipient wasn't found
        logger.warning(f"No recipient found for message: {message.header.message_id}")
        return False


class MessageProcessor:
    """
    Processes incoming messages with validation, prioritization, and error handling.
    """
    def __init__(self, router: MessageRouter):
        self.router = router
        self.priority_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        self.processing_tasks = []
        self.running = False
        
    async def start(self):
        """Start processing messages."""
        self.running = True
        
        # Create a processing task for each priority level
        for priority in MessagePriority:
            task = asyncio.create_task(self._process_queue(priority))
            self.processing_tasks.append(task)
            
        logger.info("Message processor started")
        
    async def stop(self):
        """Stop processing messages."""
        self.running = False
        
        # Wait for all tasks to complete
        for task in self.processing_tasks:
            task.cancel()
            
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        logger.info("Message processor stopped")
        
    async def enqueue_message(self, message: Message):
        """Enqueue a message for processing based on its priority."""
        priority = message.header.priority
        await self.priority_queues[priority].put(message)
        logger.debug(f"Message {message.header.message_id} enqueued with priority {priority}")
        
    async def _process_queue(self, priority: MessagePriority):
        """Process messages from a specific priority queue."""
        queue = self.priority_queues[priority]
        
        while self.running:
            try:
                # Get a message from the queue
                message = await queue.get()
                
                # Process the message
                success = await self._process_message(message)
                
                # Mark the task as done
                queue.task_done()
                
                if not success:
                    logger.warning(f"Failed to process message {message.header.message_id}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message from {priority} queue: {str(e)}")
                
    async def _process_message(self, message: Message) -> bool:
        """
        Process a single message.
        
        Returns True if the message was successfully processed, False otherwise.
        """
        try:
            # Validate the message
            if not self._validate_message(message):
                logger.warning(f"Invalid message: {message.header.message_id}")
                return False
                
            # Add processing timestamp to trace
            message.header.trace.agent_hops.append({
                "agent": "message_processor",
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "process"
            })
            
            # Route the message
            return self.router.route_message(message)
            
        except Exception as e:
            logger.error(f"Error processing message {message.header.message_id}: {str(e)}")
            
            # Try to send an error response if possible
            try:
                if message.header.route.reply_to:
                    error_response = create_error_response(
                        sender="message_processor",
                        recipient=message.header.route.reply_to,
                        error_message=f"Error processing message: {str(e)}",
                        error_code="PROCESSING_ERROR",
                        original_message_id=message.header.message_id,
                        correlation_id=message.header.correlation_id
                    )
                    await self.enqueue_message(error_response)
            except Exception as e2:
                logger.error(f"Error creating error response: {str(e2)}")
                
            return False
            
    def _validate_message(self, message: Message) -> bool:
        """Validate a message."""
        # Check for required fields
        if not message.header.message_id:
            logger.error("Message missing message_id")
            return False
            
        if not message.header.message_type:
            logger.error("Message missing message_type")
            return False
            
        if not message.header.route.sender:
            logger.error("Message missing sender")
            return False
            
        if not message.header.route.recipient and not message.header.route.broadcast:
            logger.error("Message missing recipient and not broadcast")
            return False
            
        # Additional validation could be added here
        
        return True

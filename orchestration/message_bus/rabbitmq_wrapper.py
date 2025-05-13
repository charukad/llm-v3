"""
RabbitMQ wrapper for the Multi-agent Communication Protocol (MCP).

This module provides a wrapper around the RabbitMQ client library to support
the MCP message format and provide reliable messaging between agents.
"""
import json
import asyncio
import aio_pika
from typing import Dict, Any, Optional, List, Callable, Union
import logging
import time
import ssl
from pydantic import ValidationError
from .message_formats import Message, MessageType, MessagePriority, create_error_response
from .message_handler import MessageRouter, MessageProcessor
from ..monitoring.logger import get_logger
from ..monitoring.metrics import record_message_metrics

logger = get_logger(__name__)


class RabbitMQBus:
    """
    RabbitMQ implementation of the message bus for the MCP.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        vhost: str = "/",
        username: str = "guest",
        password: str = "guest",
        use_ssl: bool = False,
        ssl_options: Dict[str, Any] = None,
        connection_attempts: int = 3,
        retry_delay: int = 5,
        heartbeat: int = 60
    ):
        self.host = host
        self.port = port
        self.vhost = vhost
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.ssl_options = ssl_options or {}
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        self.heartbeat = heartbeat
        
        self.connection = None
        self.channel = None
        self.exchange_name = "math_system"
        self.exchange_type = "topic"
        
        self.router = MessageRouter()
        self.processor = MessageProcessor(self.router)
        
        self.response_handlers: Dict[str, asyncio.Future] = {}
        self.message_listeners: Dict[str, List[Callable]] = {}
        
        # Queue to track messages being sent
        self.message_queue = asyncio.Queue()
        
    async def connect(self):
        """Connect to RabbitMQ server."""
        connection_params = {
            "host": self.host,
            "port": self.port,
            "login": self.username,
            "password": self.password,
            "virtualhost": self.vhost,
            "heartbeat": self.heartbeat
        }
        
        if self.use_ssl:
            ssl_context = ssl.create_default_context()
            for k, v in self.ssl_options.items():
                if hasattr(ssl_context, k):
                    setattr(ssl_context, k, v)
            connection_params["ssl"] = True
            connection_params["ssl_context"] = ssl_context
        
        # Try to connect with retry
        for attempt in range(1, self.connection_attempts + 1):
            try:
                self.connection = await aio_pika.connect_robust(**connection_params)
                self.channel = await self.connection.channel()
                
                # Declare exchange
                await self.channel.declare_exchange(
                    self.exchange_name,
                    self.exchange_type,
                    durable=True
                )
                
                # Start message processor
                await self.processor.start()
                
                # Start message sender task
                self._sender_task = asyncio.create_task(self._message_sender())
                
                logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}/{self.vhost}")
                return
                
            except Exception as e:
                if attempt == self.connection_attempts:
                    logger.error(f"Failed to connect to RabbitMQ after {self.connection_attempts} attempts: {str(e)}")
                    raise
                else:
                    logger.warning(f"Connection attempt {attempt} failed, retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                    
    async def disconnect(self):
        """Disconnect from RabbitMQ server."""
        # Stop the message processor
        await self.processor.stop()
        
        # Cancel the sender task
        if hasattr(self, '_sender_task'):
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
        
        # Close the channel and connection
        if self.channel:
            await self.channel.close()
            self.channel = None
            
        if self.connection:
            await self.connection.close()
            self.connection = None
            
        logger.info("Disconnected from RabbitMQ")
            
    async def _message_sender(self):
        """Background task to send messages from the queue."""
        while True:
            try:
                message, routing_key, future = await self.message_queue.get()
                
                try:
                    # Serialize the message
                    message_json = message.json()
                    
                    # Send the message
                    await self.channel.default_exchange.publish(
                        aio_pika.Message(
                            body=message_json.encode(),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                            message_id=message.header.message_id,
                            correlation_id=message.header.correlation_id,
                            priority=self._get_priority_value(message.header.priority),
                            expiration=str(message.header.route.ttl * 1000)  # Convert to milliseconds
                        ),
                        routing_key=routing_key
                    )
                    
                    # Record metrics
                    record_message_metrics(
                        message_type=message.header.message_type,
                        sender=message.header.route.sender,
                        recipient=message.header.route.recipient,
                        size=len(message_json)
                    )
                    
                    if future:
                        future.set_result(True)
                        
                except Exception as e:
                    logger.error(f"Error sending message {message.header.message_id}: {str(e)}")
                    if future:
                        future.set_exception(e)
                        
                finally:
                    self.message_queue.task_done()
                    
            except asyncio.CancelledError:
                # Task was cancelled, exit
                break
                
            except Exception as e:
                logger.error(f"Unexpected error in message sender: {str(e)}")
                await asyncio.sleep(1)  # Avoid tight loop if there's an error
                
    def _get_priority_value(self, priority: MessagePriority) -> int:
        """Convert MessagePriority enum to RabbitMQ priority value (0-9)."""
        priority_map = {
            MessagePriority.LOW: 1,
            MessagePriority.NORMAL: 5,
            MessagePriority.HIGH: 7,
            MessagePriority.CRITICAL: 9
        }
        return priority_map.get(priority, 5)
                
    async def declare_queue(self, queue_name: str, durable: bool = True, exclusive: bool = False):
        """Declare a queue and bind it to the exchange."""
        queue = await self.channel.declare_queue(
            queue_name,
            durable=durable,
            exclusive=exclusive,
            auto_delete=exclusive  # Auto-delete if exclusive
        )
        return queue
        
    async def bind_queue(self, queue_name: str, routing_key: str):
        """Bind a queue to a routing key."""
        queue = await self.channel.get_queue(queue_name)
        await queue.bind(self.exchange_name, routing_key)
        logger.debug(f"Queue {queue_name} bound to routing key {routing_key}")
        
    async def setup_agent_queues(self, agent_id: str, capabilities: List[str]):
        """Set up queues for an agent based on its capabilities."""
        # Main agent queue for direct messages
        main_queue = await self.declare_queue(f"agent.{agent_id}", durable=True)
        await self.bind_queue(f"agent.{agent_id}", f"agent.{agent_id}")
        
        # Bind to capability-based routing keys
        for capability in capabilities:
            await self.bind_queue(f"agent.{agent_id}", f"capability.{capability}")
            
        # Register agent with router
        self.router.register_agent(agent_id, capabilities)
        
        return main_queue
        
    async def setup_consumer(self, queue_name: str, callback: Callable):
        """Set up a consumer for a queue."""
        queue = await self.channel.get_queue(queue_name)
        
        async def _message_handler(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    # Parse the message
                    message_json = message.body.decode()
                    parsed_message = Message.parse_raw(message_json)
                    
                    # Process with user callback
                    await callback(parsed_message)
                    
                except ValidationError as e:
                    logger.error(f"Invalid message format: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
        
        # Start consuming
        consumer_tag = await queue.consume(_message_handler)
        logger.info(f"Consumer set up for queue {queue_name} with tag {consumer_tag}")
        return consumer_tag
        
    async def send_message(self, message: Message) -> bool:
        """Send a message to the exchange."""
        if not self.channel:
            logger.error("Not connected to RabbitMQ")
            return False
            
        # Determine routing key
        if message.header.route.broadcast:
            routing_key = f"broadcast.{message.header.message_type}"
        elif message.header.route.recipient.startswith("capability."):
            # Capability-based routing
            routing_key = message.header.route.recipient
        else:
            # Direct routing to agent
            routing_key = f"agent.{message.header.route.recipient}"
            
        # Create a future for tracking message delivery
        future = asyncio.Future()
        
        # Add to send queue
        await self.message_queue.put((message, routing_key, future))
        
        try:
            # Wait for the message to be sent
            await future
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False
            
    async def send_and_wait_response(
        self,
        message: Message,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """Send a message and wait for a response."""
        if not message.header.correlation_id:
            logger.error("Message must have a correlation_id for send_and_wait_response")
            return None
            
        # Set up response handling
        response_future = asyncio.Future()
        self.response_handlers[message.header.correlation_id] = response_future
        
        # Send the message
        sent = await self.send_message(message)
        if not sent:
            del self.response_handlers[message.header.correlation_id]
            return None
            
        try:
            # Wait for the response with timeout
            response = await asyncio.wait_for(response_future, timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response to {message.header.message_id}")
            return None
        finally:
            # Clean up
            if message.header.correlation_id in self.response_handlers:
                del self.response_handlers[message.header.correlation_id]
                
    async def handle_response(self, message: Message):
        """Handle a response message."""
        correlation_id = message.header.correlation_id
        if correlation_id in self.response_handlers:
            future = self.response_handlers[correlation_id]
            if not future.done():
                future.set_result(message)
            else:
                logger.warning(f"Response received for {correlation_id} but future already done")
        else:
            logger.warning(f"Response received for unknown correlation_id: {correlation_id}")
            
    def add_message_listener(self, message_type: MessageType, callback: Callable):
        """Add a listener for messages of a specific type."""
        if message_type not in self.message_listeners:
            self.message_listeners[message_type] = []
        self.message_listeners[message_type].append(callback)
        
    async def notify_listeners(self, message: Message):
        """Notify all listeners for a message type."""
        message_type = message.header.message_type
        if message_type in self.message_listeners:
            for callback in self.message_listeners[message_type]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in message listener for {message_type}: {str(e)}")


# Create a singleton instance
_message_bus_instance = None

def get_message_bus(config: Dict[str, Any] = None) -> RabbitMQBus:
    """Get or create the message bus singleton instance."""
    global _message_bus_instance
    if _message_bus_instance is None:
        config = config or {}
        _message_bus_instance = RabbitMQBus(**config)
    return _message_bus_instance

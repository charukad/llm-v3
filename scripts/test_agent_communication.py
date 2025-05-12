#!/usr/bin/env python3
"""
Test script for agent-to-agent communication.

This script tests the agent-to-agent communication patterns, capability
advertisement, load balancing, and fault tolerance mechanisms.
"""
import asyncio
import os
import sys
import logging
import json
import random
import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestration.agents.communication import create_agent_communication
from orchestration.agents.capability_advertisement import get_advertisement_system
from orchestration.agents.load_balancer import get_load_balancer
from orchestration.agents.fault_tolerance import get_fault_tolerance_manager
from orchestration.message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message
)
from orchestration.message_bus.rabbitmq_wrapper import get_message_bus
from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


class TestAgent:
    """Test agent implementation."""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """Initialize the test agent."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.communication = create_agent_communication(agent_id)
        self.message_bus = get_message_bus()
        self.advertisement_system = get_advertisement_system()
        self.load = 0.0
        self.running = False
        
    async def start(self):
        """Start the agent."""
        logger.info(f"Starting agent {self.agent_id}")
        
        # Connect to message bus if not already connected
        if not self.message_bus.connection:
            await self.message_bus.connect()
            
        # Setup agent queue
        queue = await self.message_bus.setup_agent_queues(self.agent_id, self.capabilities)
        
        # Setup consumer
        await self.message_bus.setup_consumer(
            f"agent.{self.agent_id}",
            self.handle_message
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
            self.handle_message
        )
        
        # Subscribe to specific message types
        await self.communication.subscribe(
            MessageType.HEARTBEAT,
            self.handle_heartbeat
        )
        
        # Advertise capabilities
        await self.communication.advertise_capabilities(
            self.capabilities,
            {
                "agent_type": self.agent_type,
                "agent_id": self.agent_id,
                "description": f"Test agent for {self.agent_type}"
            }
        )
        
        # Start heartbeat task
        self.running = True
        asyncio.create_task(self.heartbeat_task())
        
        logger.info(f"Agent {self.agent_id} started successfully")
        
    async def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")
        
    async def handle_message(self, message: Message):
        """Handle incoming messages."""
        # Log message receipt
        logger.info(f"Agent {self.agent_id} received message: {message.header.message_type}")
        
        # Pass to communication module for processing
        await self.communication.handle_message(message)
        
    async def handle_heartbeat(self, message: Message):
        """Handle heartbeat messages."""
        # Check if it's a health check
        if message.body.get("health_check", False):
            # Send a response
            response = create_message(
                message_type=MessageType.HEARTBEAT,
                sender=self.agent_id,
                recipient=message.header.route.sender,
                body={"status": "healthy", "load": self.load},
                correlation_id=message.header.correlation_id
            )
            
            await self.message_bus.send_message(response)
            
    async def heartbeat_task(self):
        """Periodically send heartbeat messages."""
        while self.running:
            # Simulate random load
            self.load = random.random() * 0.8  # 0.0 to 0.8
            
            # Send heartbeat
            await self.communication.send_heartbeat(self.load)
            
            # Wait before next heartbeat
            await asyncio.sleep(10)
            
    async def simulate_work(self, duration: float = 1.0, load_increase: float = 0.2):
        """
        Simulate work being done by the agent.
        
        Args:
            duration: Duration of the work in seconds
            load_increase: How much to increase the load during work
        """
        # Increase load
        original_load = self.load
        self.load = min(1.0, original_load + load_increase)
        
        # Simulate work
        await asyncio.sleep(duration)
        
        # Return to original load
        self.load = original_load


async def test_agent_communication():
    """Test agent-to-agent communication."""
    logger.info("Testing agent-to-agent communication...")
    
    # Create test agents
    agents = [
        TestAgent("test_llm_agent", "llm", [
            "classify_query", "generate_response", "explain_math"
        ]),
        TestAgent("test_math_agent", "computation", [
            "compute", "solve_equation", "differentiate", "integrate"
        ]),
        TestAgent("test_viz_agent", "visualization", [
            "generate_visualization", "plot_function", "plot_3d"
        ])
    ]
    
    # Start agents
    for agent in agents:
        await agent.start()
        
    # Wait for advertisements to propagate
    await asyncio.sleep(2)
    
    # Test messaging between agents
    llm_agent = agents[0].communication
    math_agent = agents[1].communication
    viz_agent = agents[2].communication
    
    # Test request-response pattern
    logger.info("Testing request-response pattern...")
    success, response = await llm_agent.request(
        recipient="test_math_agent",
        request_type=MessageType.COMPUTATION_REQUEST,
        body={
            "expression": "x^2 + 2*x + 1",
            "operation": "differentiate",
            "variable": "x"
        },
        timeout=5.0
    )
    
    logger.info(f"Request success: {success}")
    if success:
        logger.info(f"Response: {response.body}")
        
    # Test broadcast
    logger.info("Testing broadcast messaging...")
    broadcast_success = await llm_agent.broadcast(
        message_type=MessageType.STATUS_UPDATE,
        body={
            "status": "system_announcement",
            "message": "This is a test broadcast message"
        }
    )
    
    logger.info(f"Broadcast success: {broadcast_success}")
    
    # Test load balancer
    logger.info("Testing load balancer...")
    load_balancer = get_load_balancer()
    
    # Test agent selection
    for capability in ["compute", "generate_visualization", "classify_query"]:
        agent_id = load_balancer.select_agent(capability)
        logger.info(f"Selected agent for {capability}: {agent_id}")
        
    # Test fault tolerance
    logger.info("Testing fault tolerance...")
    fault_manager = get_fault_tolerance_manager()
    
    # Make resilient request
    success, response = await fault_manager.make_resilient_request(
        capability="generate_visualization",
        request_type=MessageType.VISUALIZATION_REQUEST,
        body={
            "visualization_type": "function_plot_2d",
            "expression": "sin(x)",
            "x_range": [-3.14, 3.14]
        },
        retry_policy="default",
        timeout=5.0
    )
    
    logger.info(f"Resilient request success: {success}")
    if success:
        logger.info(f"Response: {response.body}")
        
    # Test work distribution
    logger.info("Testing work distribution...")
    work_items = [
        {"id": 1, "expression": "x^2", "operation": "differentiate"},
        {"id": 2, "expression": "sin(x)", "operation": "differentiate"},
        {"id": 3, "expression": "e^x", "operation": "differentiate"},
        {"id": 4, "expression": "log(x)", "operation": "differentiate"},
        {"id": 5, "expression": "tan(x)", "operation": "differentiate"}
    ]
    
    distribution = load_balancer.distribute_work(work_items, "compute")
    
    for agent_id, items in distribution.items():
        logger.info(f"Agent {agent_id} assigned {len(items)} items: {[item['id'] for item in items]}")
        
    # Stop agents
    for agent in agents:
        await agent.stop()
        
    logger.info("Agent-to-agent communication test completed")


async def main():
    """Run the test."""
    logger.info("Starting agent communication tests...")
    
    # Connect to message bus
    message_bus = get_message_bus({
        "host": "localhost",
        "port": 5672,
        "username": "guest",
        "password": "guest"
    })
    
    try:
        # Connect to message bus
        await message_bus.connect()
        logger.info("Connected to message bus")
        
        # Run the test
        await test_agent_communication()
        
        # Disconnect from message bus
        await message_bus.disconnect()
        logger.info("Disconnected from message bus")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())

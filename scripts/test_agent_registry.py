#!/usr/bin/env python3
"""
Test script for agent registry and orchestration components.

This script tests the agent registry and basic messaging functionality.
"""
import asyncio
import os
import sys
import logging
import json
from typing import Dict, Any

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestration.agents.registry import get_agent_registry
from orchestration.message_bus.rabbitmq_wrapper import get_message_bus
from orchestration.message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message
)
from orchestration.manager.orchestration_manager import get_orchestration_manager
from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


async def test_agent_registry():
    """Test agent registry functionality."""
    logger.info("Testing agent registry...")
    
    # Get registry instance
    registry = get_agent_registry()
    
    # Get all active agents
    active_agents = registry.get_active_agents()
    logger.info(f"Active agents: {len(active_agents)}")
    
    # Get all capabilities
    capabilities = registry.get_all_capabilities()
    logger.info(f"Available capabilities: {capabilities}")
    
    # Test finding agents by capability
    for capability in capabilities:
        agents = registry.find_agents_by_capability(capability)
        logger.info(f"Agents with capability '{capability}': {agents}")
    
    # Test finding agents by type
    agent_types = set(agent["agent_type"] for agent in active_agents)
    for agent_type in agent_types:
        agents = registry.find_agents_by_type(agent_type)
        logger.info(f"Agents of type '{agent_type}': {agents}")
    
    # Test finding optimal agent
    for capability in capabilities:
        optimal_agent = registry.get_optimal_agent_for_capability(capability)
        logger.info(f"Optimal agent for '{capability}': {optimal_agent}")
    
    logger.info("Agent registry test completed successfully")


async def test_message_creation():
    """Test message creation."""
    logger.info("Testing message creation...")
    
    # Create a simple message
    message = create_message(
        message_type=MessageType.QUERY,
        sender="test_script",
        recipient="core_llm_agent",
        body={"query": "What is the derivative of x^2?"},
        priority=MessagePriority.NORMAL
    )
    
    # Log message details
    logger.info(f"Created message with ID: {message.header.message_id}")
    logger.info(f"Message JSON: {message.json(indent=2)}")
    
    logger.info("Message creation test completed successfully")
    return message


async def test_message_bus():
    """Test message bus connection and basic operations."""
    logger.info("Testing message bus...")
    
    # Get message bus instance
    message_bus = get_message_bus({
        "host": "localhost",
        "port": 5672,
        "username": "guest",
        "password": "guest"
    })
    
    try:
        # Connect to the message bus
        await message_bus.connect()
        logger.info("Connected to message bus")
        
        # Create a test message
        message = await test_message_creation()
        
        # Send the message (this will just add it to the queue since we don't have receivers)
        sent = await message_bus.send_message(message)
        logger.info(f"Message sent: {sent}")
        
        # Disconnect
        await message_bus.disconnect()
        logger.info("Disconnected from message bus")
        
    except Exception as e:
        logger.error(f"Error testing message bus: {str(e)}")
        return False
    
    logger.info("Message bus test completed successfully")
    return True


async def test_orchestration_manager():
    """Test orchestration manager functionality."""
    logger.info("Testing orchestration manager...")
    
    # Get orchestration manager instance
    orchestration_manager = get_orchestration_manager()
    
    try:
        # Initialize the orchestration manager
        await orchestration_manager.initialize()
        logger.info("Orchestration manager initialized")
        
        # List available workflows
        workflows = orchestration_manager.workflow_registry.list_workflows()
        logger.info(f"Available workflows: {workflows}")
        
        # Start a test workflow (this won't actually process since we don't have agents running)
        workflow_id, future = await orchestration_manager.start_workflow(
            workflow_type="math_problem_solving",
            initial_data={"query": "What is the derivative of x^2?"},
            conversation_id="test_conversation"
        )
        
        logger.info(f"Started workflow with ID: {workflow_id}")
        
        # Get workflow status
        workflow = await orchestration_manager.get_workflow(workflow_id)
        logger.info(f"Workflow status: {workflow.status}")
        logger.info(f"Current step: {workflow.current_step_index}/{len(workflow.steps)}")
        
        # The workflow will likely be in waiting or running state since no agents are processing messages
        
    except Exception as e:
        logger.error(f"Error testing orchestration manager: {str(e)}")
        return False
    
    logger.info("Orchestration manager test completed successfully")
    return True


async def main():
    """Run all tests."""
    logger.info("Starting agent registry and orchestration tests...")
    
    # Test agent registry
    await test_agent_registry()
    
    # Test message creation
    await test_message_creation()
    
    # Test message bus - commented out if RabbitMQ is not running
    # await test_message_bus()
    
    # Test orchestration manager - commented out if RabbitMQ is not running
    # await test_orchestration_manager()
    
    logger.info("All tests completed")


if __name__ == "__main__":
    # Run asynchronous tests
    asyncio.run(main())

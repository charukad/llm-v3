"""
Capability Advertisement System for the Mathematical Multimodal LLM System.

This module provides functionality for agents to advertise their capabilities
and for other components to discover available capabilities in the system.
"""
import asyncio
import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
import json

from ..message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message
)
from ..message_bus.rabbitmq_wrapper import get_message_bus
from ..monitoring.logger import get_logger
from .registry import get_agent_registry

logger = get_logger(__name__)


class CapabilityAdvertisement:
    """
    Capability advertisement and discovery system.
    
    This component handles the advertisement of agent capabilities and
    provides discovery mechanisms for finding agents with specific capabilities.
    """
    
    def __init__(self):
        """Initialize the capability advertisement system."""
        self.message_bus = get_message_bus()
        self.agent_registry = get_agent_registry()
        self.last_advertisement: Dict[str, Dict[str, Any]] = {}
        self.advertisement_callbacks: List[callable] = []
        
        # Register for capability advertisements
        self.message_bus.add_message_listener(
            MessageType.CAPABILITY_ADVERTISEMENT,
            self._handle_capability_advertisement
        )
        
        # Register for status updates
        self.message_bus.add_message_listener(
            MessageType.STATUS_UPDATE,
            self._handle_status_update
        )
        
        # Register for heartbeats
        self.message_bus.add_message_listener(
            MessageType.HEARTBEAT,
            self._handle_heartbeat
        )
        
    async def _handle_capability_advertisement(self, message: Message):
        """
        Handle capability advertisement messages.
        
        Args:
            message: The capability advertisement message
        """
        body = message.body
        agent_id = body.get("agent_id")
        
        if not agent_id:
            logger.warning("Received capability advertisement without agent_id")
            return
            
        capabilities = body.get("capabilities", [])
        metadata = body.get("metadata", {})
        status = body.get("status", "active")
        
        # Get the agent's type
        agent_type = metadata.get("agent_type", "unknown")
        
        # Update the registry
        if agent_id not in self.agent_registry.agents:
            # New agent registration
            logger.info(f"Registering new agent {agent_id} with capabilities: {capabilities}")
            self.agent_registry.register_agent(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                metadata=metadata
            )
        else:
            # Update existing agent
            logger.info(f"Updating agent {agent_id} capabilities: {capabilities}")
            
            # Get current registration
            current_registration = self.agent_registry.get_agent(agent_id)
            
            # Update capabilities
            self.agent_registry.agents[agent_id]["capabilities"] = capabilities
            
            # Update metadata
            self.agent_registry.agents[agent_id]["metadata"].update(metadata)
            
            # Update capability mappings
            for capability in self.agent_registry.capabilities:
                if agent_id in self.agent_registry.capabilities[capability]:
                    self.agent_registry.capabilities[capability].remove(agent_id)
                    
            for capability in capabilities:
                if capability not in self.agent_registry.capabilities:
                    self.agent_registry.capabilities[capability] = set()
                self.agent_registry.capabilities[capability].add(agent_id)
                
        # Update status
        self.agent_registry.update_agent_status(agent_id, status, metadata)
        
        # Record this advertisement
        self.last_advertisement[agent_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "capabilities": capabilities,
            "status": status,
            "metadata": metadata
        }
        
        # Notify callbacks
        for callback in self.advertisement_callbacks:
            try:
                await callback(agent_id, capabilities, status, metadata)
            except Exception as e:
                logger.error(f"Error in advertisement callback: {str(e)}")
                
    async def _handle_status_update(self, message: Message):
        """
        Handle agent status update messages.
        
        Args:
            message: The status update message
        """
        body = message.body
        agent_id = body.get("agent_id")
        
        if not agent_id:
            logger.warning("Received status update without agent_id")
            return
            
        status = body.get("status", "active")
        load = body.get("load", 0.0)
        metadata = body.get("metadata", {})
        
        # Update the registry
        self.agent_registry.update_agent_status(agent_id, status, metadata)
        
        # Update agent load if available
        if agent_id in self.agent_registry.agents:
            self.agent_registry.agents[agent_id]["load"] = load
            
    async def _handle_heartbeat(self, message: Message):
        """
        Handle agent heartbeat messages.
        
        Args:
            message: The heartbeat message
        """
        body = message.body
        agent_id = body.get("agent_id")
        
        if not agent_id:
            logger.warning("Received heartbeat without agent_id")
            return
            
        load = body.get("load", 0.0)
        
        # Update the registry with last seen timestamp
        if agent_id in self.agent_registry.agents:
            self.agent_registry.agents[agent_id]["last_seen"] = datetime.datetime.now().isoformat()
            self.agent_registry.agents[agent_id]["load"] = load
            
    async def advertise_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Advertise an agent's capabilities.
        
        Args:
            agent_id: Agent ID
            agent_type: Type of agent
            capabilities: List of capabilities
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Create advertisement message
        body = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": capabilities,
            "status": "active",
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send broadcast message
        message = create_message(
            message_type=MessageType.CAPABILITY_ADVERTISEMENT,
            sender=agent_id,
            recipient="broadcast",
            body=body,
            broadcast=True
        )
        
        return await self.message_bus.send_message(message)
        
    def add_advertisement_callback(self, callback: callable):
        """
        Add a callback to be called when an advertisement is received.
        
        Args:
            callback: Async function that takes (agent_id, capabilities, status, metadata)
        """
        self.advertisement_callbacks.append(callback)
        
    def remove_advertisement_callback(self, callback: callable):
        """
        Remove an advertisement callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self.advertisement_callbacks:
            self.advertisement_callbacks.remove(callback)
            
    async def check_agent_health(self, max_age_seconds: int = 60):
        """
        Check agent health based on heartbeats and mark inactive agents.
        
        Args:
            max_age_seconds: Maximum age in seconds before an agent is marked inactive
        """
        now = datetime.datetime.now()
        inactive_agents = []
        
        for agent_id, agent_info in self.agent_registry.agents.items():
            if agent_info["status"] in ["registered", "active", "ready"]:
                # Check last seen timestamp
                if "last_seen" in agent_info:
                    try:
                        last_seen = datetime.datetime.fromisoformat(agent_info["last_seen"])
                        age_seconds = (now - last_seen).total_seconds()
                        
                        if age_seconds > max_age_seconds:
                            logger.warning(f"Agent {agent_id} has not sent a heartbeat in {age_seconds} seconds, marking as inactive")
                            self.agent_registry.update_agent_status(agent_id, "inactive")
                            inactive_agents.append(agent_id)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error parsing last_seen for agent {agent_id}: {str(e)}")
                        
        return inactive_agents
        
    async def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """
        Get the capabilities of a specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of capabilities
        """
        agent = self.agent_registry.get_agent(agent_id)
        if not agent:
            return []
            
        return agent.get("capabilities", [])
        
    async def get_agents_with_capability(self, capability: str) -> List[str]:
        """
        Find agents that support a specific capability.
        
        Args:
            capability: Capability name
            
        Returns:
            List of agent IDs
        """
        return self.agent_registry.find_agents_by_capability(capability)


# Create singleton instance
_advertisement_system = None

def get_advertisement_system() -> CapabilityAdvertisement:
    """Get the capability advertisement system singleton."""
    global _advertisement_system
    if _advertisement_system is None:
        _advertisement_system = CapabilityAdvertisement()
    return _advertisement_system

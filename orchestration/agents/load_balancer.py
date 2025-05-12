"""
Load Balancing System for the Mathematical Multimodal LLM System.

This module provides load balancing functionality to distribute work efficiently
among agents based on their capabilities and current load.
"""
import asyncio
import random
import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
import json
import heapq

from ..message_bus.message_formats import (
    Message, MessageType, MessagePriority, create_message
)
from ..monitoring.logger import get_logger
from .registry import get_agent_registry

logger = get_logger(__name__)


class LoadBalancer:
    """
    Load balancer for distributing work among agents.
    
    This component provides mechanisms for selecting agents based on
    their capabilities and current load.
    """
    
    def __init__(self):
        """Initialize the load balancer."""
        self.agent_registry = get_agent_registry()
        self.priority_weights: Dict[str, float] = {}
        self.capability_weights: Dict[str, float] = {}
        
    def select_agent(
        self,
        capability: str,
        consider_load: bool = True,
        exclude_agents: List[str] = None
    ) -> Optional[str]:
        """
        Select an agent with the specified capability.
        
        Args:
            capability: Required capability
            consider_load: Whether to consider agent load
            exclude_agents: Agents to exclude from selection
            
        Returns:
            Selected agent ID or None if no suitable agent found
        """
        # Get all agents with the capability
        agents = self.agent_registry.find_agents_by_capability(capability)
        
        if not agents:
            return None
            
        # Filter out excluded agents
        if exclude_agents:
            agents = [a for a in agents if a not in exclude_agents]
            
        if not agents:
            return None
            
        if not consider_load:
            # If load is not a factor, choose randomly with capability weight bias
            if capability in self.capability_weights:
                # Higher weight = higher priority for the capability
                # Apply weighted random selection
                weighted_agents = []
                base_weight = 1.0
                capability_factor = self.capability_weights.get(capability, 1.0)
                
                for agent_id in agents:
                    # Higher weights make the agent more likely to be selected
                    agent_priority = self.priority_weights.get(agent_id, 1.0)
                    weight = base_weight * agent_priority * capability_factor
                    weighted_agents.append((agent_id, weight))
                    
                # Perform weighted random selection
                total_weight = sum(w for _, w in weighted_agents)
                selection = random.uniform(0, total_weight)
                current = 0
                
                for agent_id, weight in weighted_agents:
                    current += weight
                    if current >= selection:
                        return agent_id
                        
                # If we get here, return the last agent (should not happen)
                return weighted_agents[-1][0]
            else:
                # No weights, just random selection
                return random.choice(agents)
                
        # Consider load in the selection
        candidates = []
        for agent_id in agents:
            agent_info = self.agent_registry.get_agent(agent_id)
            if not agent_info:
                continue
                
            # Check if agent is in a suitable status
            if agent_info.get("status") not in ["active", "ready", "registered"]:
                continue
                
            # Get agent load
            load = agent_info.get("load", 0.0)
            
            # Get agent priority
            priority = self.priority_weights.get(agent_id, 1.0)
            
            # Get capability-specific priority
            capability_priority = self.capability_weights.get(capability, 1.0)
            
            # Compute selection score (lower is better)
            # Formula: load / (priority * capability_priority)
            # This means higher load = worse score
            # Higher priority = better score
            score = load / (priority * capability_priority)
            
            # Add to candidates
            candidates.append((score, agent_id))
            
        if not candidates:
            return None
            
        # Sort by score (lowest first)
        candidates.sort()
        
        # Return the agent with the lowest score
        return candidates[0][1]
        
    def set_agent_priority(self, agent_id: str, priority: float):
        """
        Set the priority weight for an agent.
        
        Args:
            agent_id: Agent ID
            priority: Priority weight (higher = more preferred)
        """
        self.priority_weights[agent_id] = max(0.1, priority)  # Ensure positive weight
        
    def set_capability_priority(self, capability: str, priority: float):
        """
        Set the priority weight for a capability.
        
        Args:
            capability: Capability name
            priority: Priority weight (higher = more preferred)
        """
        self.capability_weights[capability] = max(0.1, priority)  # Ensure positive weight
        
    def get_agent_load(self, agent_id: str) -> float:
        """
        Get the current load of an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent load (0.0 to 1.0) or 1.0 if agent not found
        """
        agent_info = self.agent_registry.get_agent(agent_id)
        if not agent_info:
            return 1.0
            
        return agent_info.get("load", 0.0)
        
    def get_agent_by_score(
        self,
        agents: List[str],
        strategy: str = "load",
        exclude_agents: List[str] = None
    ) -> Optional[str]:
        """
        Get the best agent based on a scoring strategy.
        
        Args:
            agents: List of agent IDs to consider
            strategy: Scoring strategy ('load', 'random', 'priority')
            exclude_agents: Agents to exclude from selection
            
        Returns:
            Selected agent ID or None if no suitable agent found
        """
        if not agents:
            return None
            
        # Filter out excluded agents
        if exclude_agents:
            agents = [a for a in agents if a not in exclude_agents]
            
        if not agents:
            return None
            
        if strategy == "random":
            return random.choice(agents)
            
        candidates = []
        
        for agent_id in agents:
            agent_info = self.agent_registry.get_agent(agent_id)
            if not agent_info:
                continue
                
            # Check if agent is in a suitable status
            if agent_info.get("status") not in ["active", "ready", "registered"]:
                continue
                
            # Get agent load
            load = agent_info.get("load", 0.0)
            
            # Get agent priority
            priority = self.priority_weights.get(agent_id, 1.0)
            
            if strategy == "load":
                # Lower load is better
                score = load
            elif strategy == "priority":
                # Higher priority is better, so use negative
                score = -priority
            else:
                # Default to combined score
                score = load / priority
                
            candidates.append((score, agent_id))
            
        if not candidates:
            return None
            
        # Sort by score (lowest first)
        candidates.sort()
        
        # Return the agent with the lowest score
        return candidates[0][1]
        
    def distribute_work(
        self,
        work_items: List[Dict[str, Any]],
        capability: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Distribute work items among agents with a capability.
        
        Args:
            work_items: List of work items to distribute
            capability: Required capability for processing
            
        Returns:
            Dictionary mapping agent IDs to lists of work items
        """
        if not work_items:
            return {}
            
        # Get all agents with the capability
        agents = self.agent_registry.find_agents_by_capability(capability)
        
        if not agents:
            logger.warning(f"No agents found with capability: {capability}")
            return {}
            
        # Filter agents by status
        active_agents = []
        for agent_id in agents:
            agent_info = self.agent_registry.get_agent(agent_id)
            if agent_info and agent_info.get("status") in ["active", "ready", "registered"]:
                active_agents.append((agent_id, agent_info.get("load", 0.0)))
                
        if not active_agents:
            logger.warning(f"No active agents found with capability: {capability}")
            return {}
            
        # If only one agent, assign all work to it
        if len(active_agents) == 1:
            return {active_agents[0][0]: work_items}
            
        # Sort agents by load (lowest first)
        active_agents.sort(key=lambda x: x[1])
        
        # Distribute work proportionally inverse to load
        total_inverse_load = sum(1.0 - load for agent_id, load in active_agents)
        
        # If all agents are fully loaded, distribute evenly
        if total_inverse_load <= 0.0:
            # Distribute work evenly
            items_per_agent = len(work_items) // len(active_agents)
            result = {}
            
            for i, (agent_id, _) in enumerate(active_agents):
                if i < len(active_agents) - 1:
                    # For all but the last agent
                    result[agent_id] = work_items[i * items_per_agent:(i + 1) * items_per_agent]
                else:
                    # Last agent gets remainder
                    result[agent_id] = work_items[i * items_per_agent:]
                    
            return result
            
        # Calculate distribution based on available capacity
        distribution = {}
        items_left = len(work_items)
        total_assigned = 0
        
        for agent_id, load in active_agents:
            # Calculate share based on inverse load
            inverse_load = 1.0 - load
            share = inverse_load / total_inverse_load
            items_to_assign = int(share * len(work_items))
            
            # Ensure at least one item if any left
            if items_to_assign == 0 and items_left > 0:
                items_to_assign = 1
                
            # Cap at remaining items
            items_to_assign = min(items_to_assign, items_left)
            
            # Assign work items
            distribution[agent_id] = work_items[total_assigned:total_assigned + items_to_assign]
            
            # Update counters
            total_assigned += items_to_assign
            items_left -= items_to_assign
            
            if items_left <= 0:
                break
                
        return distribution


# Create singleton instance
_load_balancer = None

def get_load_balancer() -> LoadBalancer:
    """Get the load balancer singleton."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer

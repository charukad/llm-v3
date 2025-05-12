"""
Agent Registry for the Mathematical Multimodal LLM System.

This module provides a registry for agents and their capabilities.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
import datetime
import json
import logging
from ..monitoring.logger import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    """Registry for agents and their capabilities."""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Set[str]] = {}
        
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        endpoint: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Register an agent with its capabilities."""
        # Create agent record
        self.agents[agent_id] = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": capabilities.copy(),
            "endpoint": endpoint,
            "metadata": metadata or {},
            "status": "registered",
            "registered_at": datetime.datetime.now().isoformat(),
            "last_seen": datetime.datetime.now().isoformat()
        }
        
        # Register capabilities
        for capability in capabilities:
            if capability not in self.capabilities:
                self.capabilities[capability] = set()
            self.capabilities[capability].add(agent_id)
            
        logger.info(f"Registered agent {agent_id} of type {agent_type} with capabilities: {capabilities}")
        
    def register_service(
        self,
        service_id: str,
        service_info: Dict[str, Any]
    ):
        """
        Register a service with the registry.
        
        Args:
            service_id: The identifier for the service
            service_info: Information about the service, including its instance
        """
        # Store service like an agent with special type
        self.agents[service_id] = {
            "agent_id": service_id,
            "agent_type": "service",
            "capabilities": [],
            "endpoint": None,
            "metadata": service_info.get("metadata", {}),
            "status": "registered",
            "registered_at": datetime.datetime.now().isoformat(),
            "last_seen": datetime.datetime.now().isoformat(),
            "instance": service_info.get("instance")
        }
        
        logger.info(f"Registered service {service_id}")
        
    def deregister_agent(self, agent_id: str):
        """Deregister an agent."""
        if agent_id not in self.agents:
            return False
            
        # Get agent capabilities
        agent_capabilities = self.agents[agent_id]["capabilities"]
        
        # Remove agent from capability mappings
        for capability in agent_capabilities:
            if capability in self.capabilities and agent_id in self.capabilities[capability]:
                self.capabilities[capability].remove(agent_id)
                if not self.capabilities[capability]:
                    del self.capabilities[capability]
                    
        # Remove agent record
        del self.agents[agent_id]
        
        logger.info(f"Deregistered agent {agent_id}")
        return True
        
    def update_agent_status(
        self,
        agent_id: str,
        status: str,
        metadata: Dict[str, Any] = None
    ):
        """Update an agent's status and metadata."""
        if agent_id not in self.agents:
            return False
            
        # Update status and last seen timestamp
        self.agents[agent_id]["status"] = status
        self.agents[agent_id]["last_seen"] = datetime.datetime.now().isoformat()
        
        # Update metadata if provided
        if metadata:
            self.agents[agent_id]["metadata"].update(metadata)
            
        return True
        
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
        
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information by ID, including instance if available."""
        agent = self.get_agent(agent_id)
        if agent and "instance" not in agent:
            agent = agent.copy()
            agent["instance"] = None
        return agent

    def get_service_info(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get service information by ID."""
        # For now, we'll use the same logic as get_agent
        return self.get_agent(service_id)

    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that support a specific capability."""
        if capability not in self.capabilities:
            return []
            
        # Filter for active agents only
        active_agents = []
        for agent_id in self.capabilities[capability]:
            if agent_id in self.agents and self.agents[agent_id]["status"] in ["registered", "active", "ready"]:
                active_agents.append(agent_id)
                
        return active_agents
        
    def find_agents_by_type(self, agent_type: str) -> List[str]:
        """Find agents of a specific type."""
        return [
            agent_id for agent_id, agent_info in self.agents.items()
            if agent_info["agent_type"] == agent_type and agent_info["status"] != "offline"
        ]
        
    def get_all_capabilities(self) -> List[str]:
        """Get all registered capabilities."""
        return list(self.capabilities.keys())
        
    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get all active agents."""
        return [
            agent_info for agent_id, agent_info in self.agents.items()
            if agent_info["status"] in ["registered", "active", "ready"]
        ]
        
    def has_capability(self, capability: str) -> bool:
        """Check if a capability is supported by any active agent."""
        if capability not in self.capabilities:
            return False
            
        # Check if any agent with this capability is active
        for agent_id in self.capabilities[capability]:
            if agent_id in self.agents and self.agents[agent_id]["status"] in ["registered", "active", "ready"]:
                return True
                
        return False
        
    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get the capabilities of a specific agent."""
        if agent_id not in self.agents:
            return []
            
        return self.agents[agent_id]["capabilities"]
        
    def get_optimal_agent_for_capability(self, capability: str) -> Optional[str]:
        """
        Get the optimal agent for a capability.
        
        Currently selects the first active agent with the capability.
        Could be extended to consider load, performance metrics, etc.
        """
        agents = self.find_agents_by_capability(capability)
        return agents[0] if agents else None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary representation."""
        return {
            "agents": {agent_id: agent_info.copy() for agent_id, agent_info in self.agents.items()},
            "capabilities": {capability: list(agents) for capability, agents in self.capabilities.items()}
        }
        
    def from_dict(self, data: Dict[str, Any]):
        """Load registry from dictionary representation."""
        self.agents = {}
        self.capabilities = {}
        
        # Load agents
        for agent_id, agent_info in data.get("agents", {}).items():
            self.agents[agent_id] = agent_info.copy()
            
        # Load capabilities
        for capability, agent_ids in data.get("capabilities", {}).items():
            self.capabilities[capability] = set(agent_ids)

    def register_agent_instance(self, agent_id: str, instance: Any):
        """
        Register an agent instance with the registry.
        
        Args:
            agent_id: The agent identifier
            instance: The agent instance
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not registered. Register the agent first.")
            return
        
        # Store the instance
        self.agents[agent_id]["instance"] = instance
        logger.info(f"Registered instance for agent {agent_id}")
        return True


# Create a singleton instance
_agent_registry_instance = None

def get_agent_registry() -> AgentRegistry:
    """Get or create the agent registry singleton instance."""
    global _agent_registry_instance
    if _agent_registry_instance is None:
        _agent_registry_instance = AgentRegistry()
    return _agent_registry_instance


# Register core system agents
def register_core_agents():
    """Register the core system agents with the registry."""
    registry = get_agent_registry()
    
    # Core LLM Agent
    registry.register_agent(
        agent_id="core_llm_agent",
        agent_type="llm",
        capabilities=[
            "classify_query",
            "generate_response",
            "explain_math",
            "translate_natural_language_to_latex",
            "generate_math_explanation",
            "analyze_math_query",
            "compute_math"
        ],
        metadata={
            "model": "mistral-7b",
            "description": "Core language model agent for natural language understanding and generation"
        }
    )
    
    # Mathematical Computation Agent
    registry.register_agent(
        agent_id="math_computation_agent",
        agent_type="computation",
        capabilities=[
            "compute",
            "solve_equation",
            "differentiate",
            "integrate",
            "linear_algebra",
            "statistics",
            "verify_solution"
        ],
        metadata={
            "engine": "sympy",
            "description": "Symbolic mathematics computation agent"
        }
    )
    
    # OCR Agent
    registry.register_agent(
        agent_id="ocr_agent",
        agent_type="ocr",
        capabilities=[
            "recognize_math",
            "process_image",
            "extract_diagram",
            "detect_structure"
        ],
        metadata={
            "engine": "paddleocr",
            "description": "Optical character recognition agent for mathematical notation"
        }
    )
    
    # Visualization Agent
    registry.register_agent(
        agent_id="visualization_agent",
        agent_type="visualization",
        capabilities=[
            "generate_visualization",
            "plot_function",
            "plot_3d",
            "statistical_visualization"
        ],
        metadata={
            "engines": ["matplotlib", "plotly"],
            "description": "Mathematical visualization agent"
        }
    )
    
    # Search Agent
    registry.register_agent(
        agent_id="search_agent",
        agent_type="search",
        capabilities=[
            "external_search",
            "knowledge_retrieval",
            "citation_generation"
        ],
        metadata={
            "sources": ["google", "arxiv", "wolfram_alpha"],
            "description": "External search and knowledge retrieval agent"
        }
    )
    
    logger.info("Core system agents registered")


# Auto-register core agents when module is imported
register_core_agents()

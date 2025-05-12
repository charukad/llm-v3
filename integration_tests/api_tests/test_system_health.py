"""
System-wide health check tests.
Verifies that all critical system components are operational.
"""
import pytest
import requests
import time
from math_llm_system.orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus
from math_llm_system.orchestration.agents.registry import AgentRegistry
from math_llm_system.database.access.mongodb_wrapper import MongoDBWrapper

class TestSystemHealth:
    """Test class for system-wide health checks."""
    
    @classmethod
    def setup_class(cls):
        """Set up test resources."""
        # Base URL for API
        cls.base_url = "http://localhost:8000/api"
        
        # Initialize core components for direct testing
        cls.message_bus = RabbitMQBus()
        cls.agent_registry = AgentRegistry()
        cls.mongodb = MongoDBWrapper()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        cls.message_bus.close_connection()
    
    def test_api_health(self):
        """Test API health endpoint."""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        health_data = response.json()
        
        assert health_data["status"] == "healthy"
        assert "version" in health_data
        assert "uptime" in health_data
    
    def test_database_connectivity(self):
        """Test database connection."""
        # Direct connection test
        assert self.mongodb.get_client() is not None
        
        # API-level test
        response = requests.get(f"{self.base_url}/health/database")
        assert response.status_code == 200
        db_health = response.json()
        
        assert db_health["status"] == "connected"
        assert "ping_time_ms" in db_health
    
    def test_message_bus_connectivity(self):
        """Test message bus connection."""
        # Direct connection test
        assert self.message_bus.is_connected()
        
        # API-level test
        response = requests.get(f"{self.base_url}/health/message-bus")
        assert response.status_code == 200
        bus_health = response.json()
        
        assert bus_health["status"] == "connected"
        assert "queues" in bus_health
    
    def test_agent_availability(self):
        """Test that all required agents are available."""
        # Get registered agents
        agents = self.agent_registry.get_all_agents()
        
        # Check required agents
        required_agents = [
            "core_llm_agent",
            "math_computation_agent",
            "handwriting_recognition_agent",
            "visualization_agent",
            "search_agent"
        ]
        
        registered_agent_ids = [agent["id"] for agent in agents]
        for required_agent in required_agents:
            assert required_agent in registered_agent_ids, f"Required agent '{required_agent}' is not registered"
        
        # API-level test
        response = requests.get(f"{self.base_url}/health/agents")
        assert response.status_code == 200
        agents_health = response.json()
        
        assert agents_health["status"] == "healthy"
        assert "count" in agents_health
        assert agents_health["count"] >= len(required_agents)
        
        # Verify each required agent is healthy
        for required_agent in required_agents:
            assert required_agent in agents_health["agents"]
            assert agents_health["agents"][required_agent]["status"] == "healthy"
    
    def test_model_loading(self):
        """Test that the core model is properly loaded."""
        response = requests.get(f"{self.base_url}/health/model")
        assert response.status_code == 200
        model_health = response.json()
        
        assert model_health["status"] == "loaded"
        assert "name" in model_health
        assert "Mistral" in model_health["name"]
        assert "load_time_ms" in model_health
    
    def test_simple_end_to_end_request(self):
        """Test a simple end-to-end request to ensure the system is functioning."""
        simple_query = "What is 2+2?"
        
        response = requests.post(
            f"{self.base_url}/math/query",
            json={"query": simple_query}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "response" in result
        assert "4" in result["response"]
        assert "workflow_id" in result
        
        # Verify workflow completed successfully
        workflow_id = result["workflow_id"]
        
        # Poll for completion (with timeout)
        max_wait_time = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{self.base_url}/workflows/{workflow_id}/status")
            assert status_response.status_code == 200
            
            status = status_response.json()
            if status["status"] in ["completed", "error"]:
                break
                
            time.sleep(0.5)
        
        assert status["status"] == "completed"

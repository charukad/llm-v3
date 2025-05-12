"""
Integration tests for error handling across components.
Verifies that errors are properly handled and recovered from.
"""
import pytest
import time
import requests
from math_llm_system.orchestration.manager.orchestration_manager import OrchestrationManager
from math_llm_system.orchestration.agents.registry import AgentRegistry
from math_llm_system.orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus
from math_llm_system.database.access.conversation_repository import ConversationRepository
from math_llm_system.database.access.mongodb_wrapper import MongoDBWrapper

class TestErrorHandling:
    """Test class for error handling across components."""
    
    @classmethod
    def setup_class(cls):
        """Initialize test environment."""
        # Initialize MongoDB connection
        mongodb = MongoDBWrapper()
        cls.conversation_repo = ConversationRepository(mongodb)
        
        # Initialize message bus
        cls.message_bus = RabbitMQBus()
        
        # Initialize agent registry
        cls.agent_registry = AgentRegistry()
        
        # Initialize orchestration manager
        cls.orchestration_manager = OrchestrationManager(cls.message_bus, cls.agent_registry)
        
        # Create test conversation
        cls.test_user_id = "test_user_error_handling"
        cls.test_conversation_id = cls.conversation_repo.create_conversation(
            cls.test_user_id, 
            "Error Handling Test Conversation"
        )
        
        # API base URL
        cls.base_url = "http://localhost:8000/api"
    
    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        # Delete test conversation
        cls.conversation_repo.delete_conversation(cls.test_conversation_id)
        
        # Close connections
        cls.message_bus.close_connection()
    
    def test_invalid_mathematical_expression(self):
        """Test handling of invalid mathematical expressions."""
        # Query with invalid mathematical expression
        query = "Solve the equation x^2 + + = 0"
        
        # Start workflow
        workflow_id = self.orchestration_manager.start_workflow(
            workflow_type="math_problem_solving",
            initial_data={
                "query": query,
                "conversation_id": self.test_conversation_id
            }
        )
        
        # Wait for workflow to complete (with timeout)
        max_wait_time = 15  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            workflow_status = self.orchestration_manager.get_workflow_status(workflow_id)
            if workflow_status["state"] in ["completed", "error"]:
                break
            time.sleep(0.5)
        
        # We expect the workflow to complete (not error) but with a handled invalid expression
        assert workflow_status["state"] == "completed"
        
        # Get the result
        result = self.orchestration_manager.get_workflow_result(workflow_id)
        
        # Check that the response includes an indication of the invalid expression
        assert "invalid" in result["response"].lower() or \
               "unable to parse" in result["response"].lower() or \
               "error in the equation" in result["response"].lower()
    
    def test_timeout_recovery(self):
        """Test recovery from timeout in one agent."""
        # This test requires an agent simulator that can be instructed to time out
        # For this test, we'll use an API endpoint that simulates a timeout
        
        response = requests.post(
            f"{self.base_url}/test/simulate-timeout",
            json={
                "agent_id": "math_computation_agent",
                "timeout_seconds": 3,
                "query": "Compute the integral of x^10000 dx"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify that a workflow ID was returned
        assert "workflow_id" in result
        workflow_id = result["workflow_id"]
        
        # Poll for completion (with timeout)
        max_wait_time = 20  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{self.base_url}/workflows/{workflow_id}/status")
            assert status_response.status_code == 200
            
            status = status_response.json()
            if status["status"] in ["completed", "error"]:
                break
                
            time.sleep(0.5)
        
        # The workflow should complete even with a timeout in one agent
        assert status["status"] == "completed"
        
        # Get final result
        result_response = requests.get(f"{self.base_url}/workflows/{workflow_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        
        # The response should indicate a timeout occurred but still provide a useful response
        assert "x^10001/(10001)" in result["response"] or \
               "x^{10001}/10001" in str(result.get("latex_expressions", []))
    
    def test_agent_failure_recovery(self):
        """Test recovery from agent failure."""
        # Simulate a complete failure in an agent
        response = requests.post(
            f"{self.base_url}/test/simulate-failure",
            json={
                "agent_id": "visualization_agent",
                "query": "Plot sin(x) from 0 to 2π"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify that a workflow ID was returned
        assert "workflow_id" in result
        workflow_id = result["workflow_id"]
        
        # Poll for completion (with timeout)
        max_wait_time = 20  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{self.base_url}/workflows/{workflow_id}/status")
            assert status_response.status_code == 200
            
            status = status_response.json()
            if status["status"] in ["completed", "error"]:
                break
                
            time.sleep(0.5)
        
        # The workflow should still complete despite the agent failure
        assert status["status"] == "completed"
        
        # Get final result
        result_response = requests.get(f"{self.base_url}/workflows/{workflow_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        
        # The response should acknowledge the visualization failure but still provide a textual response
        assert "sin(x)" in result["response"]
        assert result.get("visualization_available", True) is False or \
               len(result.get("visualizations", [])) == 0
    
    def test_database_error_recovery(self):
        """Test recovery from temporary database errors."""
        # This test requires a way to simulate a temporary database failure
        # We'll use a special API endpoint for this simulation
        
        response = requests.post(
            f"{self.base_url}/test/simulate-db-error",
            json={
                "error_type": "temporary_unavailable",
                "duration_seconds": 2,
                "query": "What is the derivative of ln(x)?"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify that a workflow ID was returned
        assert "workflow_id" in result
        workflow_id = result["workflow_id"]
        
        # Poll for completion (with timeout)
        max_wait_time = 20  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{self.base_url}/workflows/{workflow_id}/status")
            
            # The status endpoint might briefly return an error during the simulated outage
            if status_response.status_code != 200:
                time.sleep(0.5)
                continue
                
            status = status_response.json()
            if status["status"] in ["completed", "error"]:
                break
                
            time.sleep(0.5)
        
        # The workflow should eventually complete as system recovers
        assert status["status"] == "completed"
        
        # Get final result
        result_response = requests.get(f"{self.base_url}/workflows/{workflow_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        
        # The response should contain the correct answer despite the temporary DB issue
        assert "1/x" in result["response"] or \
               "\\frac{1}{x}" in str(result.get("latex_expressions", []))

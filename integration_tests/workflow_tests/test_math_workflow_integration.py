"""
Comprehensive integration test for full mathematical workflow.
Tests the end-to-end process from query to response, including all agent interactions.
"""
import pytest
import json
import time
from math_llm_system.orchestration.manager.orchestration_manager import OrchestrationManager
from math_llm_system.orchestration.agents.registry import AgentRegistry
from math_llm_system.orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus
from math_llm_system.database.access.conversation_repository import ConversationRepository
from math_llm_system.database.access.mongodb_wrapper import MongoDBWrapper

class TestMathWorkflowIntegration:
    """Test class for comprehensive mathematical workflow integration tests."""
    
    @classmethod
    def setup_class(cls):
        """Initialize test environment with all necessary components."""
        # Initialize MongoDB connection
        mongodb = MongoDBWrapper()
        cls.conversation_repo = ConversationRepository(mongodb)
        
        # Initialize message bus
        cls.message_bus = RabbitMQBus()
        
        # Initialize agent registry with all agents
        cls.agent_registry = AgentRegistry()
        
        # Initialize orchestration manager
        cls.orchestration_manager = OrchestrationManager(cls.message_bus, cls.agent_registry)
        
        # Create test conversation
        cls.test_user_id = "test_user_integration"
        cls.test_conversation_id = cls.conversation_repo.create_conversation(
            cls.test_user_id, 
            "Integration Test Conversation"
        )
    
    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        # Delete test conversation
        cls.conversation_repo.delete_conversation(cls.test_conversation_id)
        
        # Close connections
        cls.message_bus.close_connection()
    
    @pytest.mark.parametrize("query,expected_elements", [
        (
            "Differentiate x^2 * sin(x) with respect to x", 
            ["2x * sin(x)", "x^2 * cos(x)", "product rule"]
        ),
        (
            "Solve the equation 2x^2 - 8 = 0 for x",
            ["x = 2", "x = -2", "square root"]
        ),
        (
            "Calculate the integral of x^3 from 0 to 1",
            ["1/4", "x^4/4", "definite integral"]
        ),
        (
            "Find the eigenvalues of matrix [[1, 2], [2, 1]]",
            ["3", "-1", "eigenvalues"]
        ),
    ])
    def test_mathematical_workflow(self, query, expected_elements):
        """
        Test end-to-end mathematical workflow with various queries.
        
        Args:
            query: Mathematical query to process
            expected_elements: Elements that should appear in the response
        """
        # Create workflow for processing the query
        workflow_id = self.orchestration_manager.start_workflow(
            workflow_type="math_problem_solving",
            initial_data={
                "query": query,
                "conversation_id": self.test_conversation_id,
                "require_visualization": True,
                "require_steps": True
            }
        )
        
        # Wait for workflow to complete (with timeout)
        max_wait_time = 30  # seconds
        start_time = time.time()
        workflow_complete = False
        
        while time.time() - start_time < max_wait_time:
            workflow_status = self.orchestration_manager.get_workflow_status(workflow_id)
            if workflow_status["state"] in ["completed", "error"]:
                workflow_complete = True
                break
            time.sleep(0.5)
        
        assert workflow_complete, f"Workflow for query '{query}' did not complete within timeout"
        assert workflow_status["state"] == "completed", f"Workflow failed with error: {workflow_status.get('error')}"
        
        # Get the workflow result
        result = self.orchestration_manager.get_workflow_result(workflow_id)
        
        # Check response contains expected mathematical content
        for element in expected_elements:
            assert element.lower() in result["response"].lower() or \
                   element.lower() in str(result.get("latex_expressions", [])).lower(), \
                   f"Expected element '{element}' not found in response for query '{query}'"
        
        # Verify agent participation by checking all expected agents were involved
        involved_agents = result.get("involved_agents", [])
        expected_agents = ["core_llm_agent", "math_computation_agent", "visualization_agent"]
        
        for agent in expected_agents:
            assert agent in involved_agents, f"Expected agent '{agent}' was not involved in workflow"
        
        # Verify visualization was generated if requested
        if result.get("require_visualization", False):
            assert "visualizations" in result and len(result["visualizations"]) > 0, \
                   "Visualization was requested but not included in result"
        
        # Verify step-by-step solution was generated if requested
        if result.get("require_steps", False):
            assert "steps" in result and len(result["steps"]) > 0, \
                   "Step-by-step solution was requested but not included in result"

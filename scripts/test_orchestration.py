"""
Test script for the orchestration manager.

This script tests the basic functionality of the orchestration manager and workflows.
"""

import logging
import sys
import os
import time
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockRabbitMQBus:
    """Mock implementation of RabbitMQBus for testing."""
    
    def __init__(self):
        """Initialize the mock message bus."""
        self.messages = []
        self.responses = {}
    
    def send_message(self, recipient, message_body, message_type, sender="test"):
        """
        Send a message.
        
        Args:
            recipient: Recipient agent ID
            message_body: Message content
            message_type: Type of message
            sender: Sender agent ID
            
        Returns:
            Message ID
        """
        message_id = f"msg_{len(self.messages)}"
        
        message = {
            "header": {
                "message_id": message_id,
                "timestamp": datetime.now().isoformat(),
                "sender": sender,
                "recipient": recipient,
                "message_type": message_type
            },
            "body": message_body
        }
        
        self.messages.append(message)
        logger.info(f"Mock message sent: {message_id} to {recipient}")
        
        return message_id
    
    def send_request_sync(self, recipient, message_body, message_type, sender="test", timeout=5):
        """
        Send a request and return a mock response.
        
        Args:
            recipient: Recipient agent ID
            message_body: Message content
            message_type: Type of message
            sender: Sender agent ID
            timeout: Request timeout
            
        Returns:
            Mock response
        """
        message_id = self.send_message(recipient, message_body, message_type, sender)
        
        # Check if we have a prepared response for this message type
        if message_type in self.responses:
            logger.info(f"Returning prepared response for message type: {message_type}")
            response = self.responses[message_type]
            
            # Add correlation ID and in_response_to
            response["header"]["correlation_id"] = message_id
            response["header"]["in_response_to"] = message_id
            
            return response
        
        # Default mock response
        return {
            "header": {
                "message_id": f"resp_{message_id}",
                "correlation_id": message_id,
                "timestamp": datetime.now().isoformat(),
                "sender": recipient,
                "recipient": sender,
                "message_type": "response",
                "in_response_to": message_id
            },
            "body": {
                "data": {
                    "result": f"Mock result for {message_type}",
                    "success": True
                }
            }
        }
    
    def register_response(self, message_type, response):
        """
        Register a prepared response for a message type.
        
        Args:
            message_type: Type of message to respond to
            response: Response to return
        """
        self.responses[message_type] = response

class MockAgentRegistry:
    """Mock implementation of AgentRegistry for testing."""
    
    def __init__(self):
        """Initialize the mock agent registry."""
        self.agents = {}
        self.capabilities = {}
    
    def add_agent(self, agent_id, agent_type, capabilities):
        """
        Add a mock agent to the registry.
        
        Args:
            agent_id: Agent ID
            agent_type: Agent type
            capabilities: List of capabilities
        """
        self.agents[agent_id] = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": capabilities,
            "status": "active"
        }
        
        for capability in capabilities:
            if capability not in self.capabilities:
                self.capabilities[capability] = []
            self.capabilities[capability].append(agent_id)
    
    def find_agents_by_capability(self, capability):
        """
        Find agents with a specific capability.
        
        Args:
            capability: Capability to look for
            
        Returns:
            List of agent information with the capability
        """
        agent_ids = self.capabilities.get(capability, [])
        
        # Convert to AgentInfo-like objects
        return [MockAgentInfo(self.agents[agent_id]) for agent_id in agent_ids 
                if agent_id in self.agents and self.agents[agent_id]["status"] == "active"]

class MockAgentInfo:
    """Mock implementation of AgentInfo for testing."""
    
    def __init__(self, data):
        """
        Initialize from data dictionary.
        
        Args:
            data: Agent data dictionary
        """
        self.agent_id = data["agent_id"]
        self.agent_type = data["agent_type"]
        self.capabilities = data["capabilities"]
        self.status = data["status"]

def test_workflow_definition():
    """Test workflow definition creation and manipulation."""
    logger.info("Testing workflow definition...")
    
    try:
        from orchestration.workflow.workflow_definition import WorkflowStep, WorkflowDefinition
        
        # Create steps
        step1 = WorkflowStep(
            step_id="step1",
            step_name="First Step",
            agent_type="test_agent",
            capability="test_capability",
            description="Test description",
            required_inputs=["input1"]
        )
        
        step2 = WorkflowStep(
            step_id="step2",
            step_name="Second Step",
            agent_type="test_agent",
            capability="another_capability",
            description="Another description",
            required_inputs=["input2"],
            outputs=["output1"]
        )
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id="test_workflow",
            workflow_name="Test Workflow",
            description="Test workflow description",
            steps=[step1, step2]
        )
        
        # Test to_dict and from_dict
        workflow_dict = workflow.to_dict()
        logger.info(f"Workflow dict: {json.dumps(workflow_dict, indent=2)}")
        
        recreated = WorkflowDefinition.from_dict(workflow_dict)
        
        # Test step retrieval
        step = workflow.get_step(0)
        logger.info(f"Step at index 0: {step.step_name}")
        
        # Test next step calculation
        next_index = workflow.get_next_step_index(0, {})
        logger.info(f"Next step index after 0: {next_index}")
        
        last_index = workflow.get_next_step_index(1, {})
        logger.info(f"Next step index after 1: {last_index}")
        
        logger.info("Workflow definition test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Workflow definition test failed: {e}")
        return False

def test_standard_workflows():
    """Test standard workflow definitions."""
    logger.info("Testing standard workflows...")
    
    try:
        from orchestration.workflow.standard_workflows import (
            get_workflow_definition,
            MATH_PROBLEM_SOLVING_WORKFLOW,
            HANDWRITING_RECOGNITION_WORKFLOW,
            KNOWLEDGE_SEARCH_WORKFLOW
        )
        
        # Test math problem solving workflow
        logger.info(f"Math problem solving workflow: {MATH_PROBLEM_SOLVING_WORKFLOW.workflow_name}")
        logger.info(f"Number of steps: {len(MATH_PROBLEM_SOLVING_WORKFLOW.steps)}")
        
        # Test handwriting recognition workflow
        logger.info(f"Handwriting recognition workflow: {HANDWRITING_RECOGNITION_WORKFLOW.workflow_name}")
        logger.info(f"Number of steps: {len(HANDWRITING_RECOGNITION_WORKFLOW.steps)}")
        
        # Test knowledge search workflow
        logger.info(f"Knowledge search workflow: {KNOWLEDGE_SEARCH_WORKFLOW.workflow_name}")
        logger.info(f"Number of steps: {len(KNOWLEDGE_SEARCH_WORKFLOW.steps)}")
        
        # Test workflow retrieval
        retrieved = get_workflow_definition("math_problem_solving")
        logger.info(f"Retrieved workflow: {retrieved.workflow_name}")
        
        # Test non-existent workflow
        nonexistent = get_workflow_definition("nonexistent")
        logger.info(f"Non-existent workflow: {nonexistent}")
        
        logger.info("Standard workflows test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Standard workflows test failed: {e}")
        return False

def test_orchestration_manager():
    """Test the orchestration manager."""
    logger.info("Testing orchestration manager...")
    
    try:
        from orchestration.manager.orchestration_manager import OrchestrationManager
        
        # Create mock components
        message_bus = MockRabbitMQBus()
        agent_registry = MockAgentRegistry()
        
        # Add mock agents
        agent_registry.add_agent(
            agent_id="test_llm_agent",
            agent_type="core_llm",
            capabilities=["classify_query", "format_response"]
        )
        
        agent_registry.add_agent(
            agent_id="test_math_agent",
            agent_type="math_computation",
            capabilities=["compute", "generate_steps"]
        )
        
        agent_registry.add_agent(
            agent_id="test_viz_agent",
            agent_type="visualization",
            capabilities=["generate_plot"]
        )
        
        # Register mock responses for specific capabilities
        message_bus.register_response("classify_query", {
            "header": {
                "message_id": "mock_classify_response",
                "sender": "test_llm_agent",
                "recipient": "orchestration_manager",
                "message_type": "response"
            },
            "body": {
                "data": {
                    "domain": "algebra",
                    "parsed_query": "x^2 - 5x + 6 = 0",
                    "operation": "solve"
                }
            }
        })
        
        message_bus.register_response("compute", {
            "header": {
                "message_id": "mock_compute_response",
                "sender": "test_math_agent",
                "recipient": "orchestration_manager",
                "message_type": "response"
            },
            "body": {
                "data": {
                    "result": "x = 2, x = 3",
                    "symbolic_result": "[2, 3]"
                }
            }
        })
        
        # Create orchestration manager
        manager = OrchestrationManager(message_bus, agent_registry)
        
        # Test workflow start
        notification_log = []
        
        def mock_notify(client_id, notification):
            """Mock notification callback."""
            notification_log.append(notification)
            logger.info(f"Notification received: {notification['event_type']}")
        
        # Start a workflow
        workflow_id = manager.start_workflow(
            workflow_type="math_problem_solving",
            initial_data={
                "query": "Solve x^2 - 5x + 6 = 0",
                "input1": "test_value",  # For required inputs in first step
                "input2": "test_value"   # For required inputs in second step
            }
        )
        
        logger.info(f"Started workflow: {workflow_id}")
        
        # Subscribe to workflow
        manager.subscribe_to_workflow(workflow_id, "test_client", mock_notify)
        
        # Get workflow state
        workflow = manager.get_workflow(workflow_id)
        logger.info(f"Workflow state: {workflow['state']}")
        
        # If workflow completed, get results
        if workflow["state"] == "completed":
            results = manager.get_workflow_results(workflow_id)
            logger.info(f"Workflow results: {results}")
        
        logger.info("Orchestration manager test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Orchestration manager test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting orchestration tests...")
    
    # Run tests
    workflow_def_success = test_workflow_definition()
    logger.info("-----------------------------------")
    
    standard_workflows_success = test_standard_workflows()
    logger.info("-----------------------------------")
    
    orchestration_success = test_orchestration_manager()
    
    # Report results
    logger.info("===================================")
    logger.info("Test Results:")
    logger.info(f"Workflow Definition: {'PASSED' if workflow_def_success else 'FAILED'}")
    logger.info(f"Standard Workflows: {'PASSED' if standard_workflows_success else 'FAILED'}")
    logger.info(f"Orchestration Manager: {'PASSED' if orchestration_success else 'FAILED'}")
    
    if workflow_def_success and standard_workflows_success and orchestration_success:
        logger.info("All orchestration tests completed successfully")
        sys.exit(0)
    else:
        logger.error("One or more orchestration tests failed")
        sys.exit(1)

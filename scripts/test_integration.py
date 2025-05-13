#!/usr/bin/env python3
"""
Integration test script for the Mathematical Multimodal LLM System.

This script tests the end-to-end functionality of the system, including:
- Mistral 7B integration
- MongoDB connections
- Message bus communication
- Agent registration and discovery
- Workflow orchestration
"""
import os
import sys
import logging
import argparse
import json
import time
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from core.agent.llm_agent import CoreLLMAgent
from core.mistral.downloader import ModelDownloader
from core.mistral.quantization import ModelQuantizer
from core.evaluation.math_evaluator import MathEvaluator

# Import database components
from database.access.mongodb_wrapper import MongoDBWrapper
from database.access.conversation_repository import ConversationRepository
from database.access.expression_repository import ExpressionRepository
from database.access.model_repository import ModelRepository

# Import orchestration components
from orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus
from orchestration.agents.registry import AgentRegistry
from orchestration.workflow.workflow_definition import WorkflowDefinition, WorkflowStep
from orchestration.manager.orchestration_manager import OrchestrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_integration.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Integration test for Mathematical Multimodal LLM System')
    
    parser.add_argument('--mongodb-uri', type=str, default='mongodb://localhost:27017',
                        help='MongoDB connection URI')
    parser.add_argument('--rabbitmq-host', type=str, default='localhost',
                        help='RabbitMQ host')
    parser.add_argument('--rabbitmq-port', type=int, default=5672,
                        help='RabbitMQ port')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Model cache directory')
    parser.add_argument('--quantization', type=str, default='4bit',
                        choices=['4bit', '8bit', 'gptq', 'none'],
                        help='Quantization method to use')
    parser.add_argument('--use-vllm', action='store_true',
                        help='Use vLLM for inference if available')
    parser.add_argument('--test-mode', choices=['basic', 'model', 'db', 'message', 'agent', 'workflow', 'all'],
                        default='basic',
                        help='Test mode to run')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='Directory to save test results')
    
    return parser.parse_args()

def test_core_llm_agent(cache_dir: Optional[str] = None, quantization: str = '4bit', use_vllm: bool = False) -> bool:
    """
    Test the Core LLM Agent with Mistral 7B.
    
    Args:
        cache_dir: Model cache directory
        quantization: Quantization method
        use_vllm: Whether to use vLLM for inference
        
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing Core LLM Agent with Mistral 7B...")
    
    try:
        # Configure agent
        agent_config = {
            "model_cache_dir": cache_dir,
            "use_quantization": quantization != "none",
            "quantization_method": quantization if quantization != "none" else None,
            "use_vllm": use_vllm,
            "temperature": 0.1,
            "max_tokens": 2048
        }
        
        # Initialize agent
        agent = CoreLLMAgent(config=agent_config)
        
        # Test 1: Generate a response
        query = "Solve the equation: 3x + 2 = 8"
        
        classification = agent.classify_mathematical_domain(query)
        logger.info(f"Domain classification: {classification['domain']} with confidence {classification['confidence']:.2f}")
        
        response = agent.generate_response(
            prompt=query,
            domain=classification["domain"],
            use_cot=True,
            num_examples=1
        )
        
        logger.info(f"Query: {query}")
        logger.info(f"Response snippet: {response[:100]}...")
        
        # Test 2: Mathematical domain classification
        calculus_query = "Find the derivative of f(x) = x^3 - 2x^2 + 5x - 3"
        calculus_classification = agent.classify_mathematical_domain(calculus_query)
        
        logger.info(f"Calculus query classification: {calculus_classification['domain']} with confidence {calculus_classification['confidence']:.2f}")
        
        # Test 3: Expression parsing
        expression = agent.parse_mathematical_expression("Solve for x: 2x + 3 = 7")
        logger.info(f"Parsed expression: {expression}")
        
        logger.info("Core LLM Agent tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Core LLM Agent test failed: {e}")
        return False

def test_database_connections(mongodb_uri: str) -> bool:
    """
    Test database connections.
    
    Args:
        mongodb_uri: MongoDB connection URI
        
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing database connections...")
    
    try:
        # Test MongoDB wrapper
        mongodb_wrapper = MongoDBWrapper(mongodb_uri)
        db_info = mongodb_wrapper.get_database_info()
        logger.info(f"Connected to MongoDB: {db_info}")
        
        # Test conversation repository
        conversation_repo = ConversationRepository(mongodb_uri)
        
        # Create a test conversation
        user_id = "test_user"
        title = "Test Conversation"
        conversation_id = conversation_repo.create_conversation(user_id, title)
        
        logger.info(f"Created test conversation with ID: {conversation_id}")
        
        # Add an interaction
        interaction = {
            "user_input": {
                "type": "text",
                "content": "What is the derivative of x^2?"
            },
            "system_response": {
                "type": "text",
                "content": "The derivative of x^2 is 2x."
            }
        }
        
        success = conversation_repo.add_interaction(conversation_id, interaction)
        if not success:
            logger.error("Failed to add interaction to conversation")
            return False
        
        # Retrieve the conversation
        conversation = conversation_repo.get_conversation(conversation_id)
        if not conversation or "interactions" not in conversation or len(conversation["interactions"]) != 1:
            logger.error("Failed to retrieve conversation or interactions")
            return False
        
        logger.info(f"Retrieved conversation: {conversation['title']} with {len(conversation['interactions'])} interactions")
        
        # Test expression repository
        expression_repo = ExpressionRepository(mongodb_uri)
        
        # Create a test expression
        expression_data = {
            "latex_representation": "\\frac{d}{dx}(x^2) = 2x",
            "symbolic_representation": "Derivative(x**2, x) = 2*x",
            "domain": "calculus",
            "conversation_id": conversation_id,
            "interaction_id": conversation["interactions"][0]["_id"]
        }
        
        expression_id = expression_repo.create_expression(expression_data)
        logger.info(f"Created test expression with ID: {expression_id}")
        
        # Search for expressions
        expressions = expression_repo.search_expressions_by_domain("calculus")
        if not expressions or len(expressions) < 1:
            logger.error("Failed to search expressions by domain")
            return False
        
        logger.info(f"Found {len(expressions)} calculus expressions")
        
        # Test model repository
        model_repo = ModelRepository(mongodb_uri)
        
        # Create a test model
        model_id = model_repo.create_model(
            name="Mistral-7B-Test",
            description="Test Mistral 7B model",
            model_type="llm"
        )
        
        logger.info(f"Created test model with ID: {model_id}")
        
        # Create a model version
        version_id = model_repo.create_model_version(
            model_id=model_id,
            version="v0.1",
            source_url="https://huggingface.co/mistralai/Mistral-7B-v0.1",
            status="downloaded"
        )
        
        logger.info(f"Created test model version with ID: {version_id}")
        
        # Create a model config
        config_id = model_repo.create_model_config(
            model_id=model_id,
            version_id=version_id,
            config={
                "quantization": "4bit",
                "use_vllm": True,
                "temperature": 0.1
            },
            name="Default Config"
        )
        
        logger.info(f"Created test model config with ID: {config_id}")
        
        # Record model metrics
        metrics_id = model_repo.record_model_metrics(
            model_id=model_id,
            version_id=version_id,
            metrics={
                "accuracy": 0.92,
                "latency": 1500
            },
            domain="algebra"
        )
        
        logger.info(f"Recorded test model metrics with ID: {metrics_id}")
        
        logger.info("Database connection tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def test_message_bus(rabbitmq_host: str, rabbitmq_port: int) -> bool:
    """
    Test message bus communication.
    
    Args:
        rabbitmq_host: RabbitMQ host
        rabbitmq_port: RabbitMQ port
        
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing message bus communication...")
    
    try:
        # Initialize message bus
        message_bus = RabbitMQBus(
            host=rabbitmq_host,
            port=rabbitmq_port
        )
        
        # Test message queue creation
        test_queue = "test_queue"
        message_bus.channel.queue_declare(queue=test_queue, durable=True)
        
        # Define message callback
        message_received = {"status": False, "content": None}
        
        def on_message_callback(ch, method, properties, body):
            message_received["status"] = True
            message_received["content"] = json.loads(body)
            logger.info(f"Received message: {message_received['content']}")
        
        # Set up consumer
        message_bus.channel.basic_consume(
            queue=test_queue,
            on_message_callback=on_message_callback,
            auto_ack=True
        )
        
        # Send a test message
        test_message = {
            "test_key": "test_value",
            "timestamp": int(time.time() * 1000)
        }
        
        message_bus.channel.basic_publish(
            exchange='',
            routing_key=test_queue,
            body=json.dumps(test_message)
        )
        
        logger.info(f"Sent test message: {test_message}")
        
        # Process message (non-blocking)
        deadline = time.time() + 5  # 5 second timeout
        while not message_received["status"] and time.time() < deadline:
            message_bus.connection.process_data_events(time_limit=0.1)
        
        # Check if message was received
        if not message_received["status"]:
            logger.error("Failed to receive test message")
            return False
        
        logger.info("Message bus communication test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Message bus communication test failed: {e}")
        return False

def test_agent_registry() -> bool:
    """
    Test agent registry functionality.
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing agent registry...")
    
    try:
        # Initialize agent registry
        registry = AgentRegistry()
        
        # Register a test agent
        agent_info = {
            "id": "test_agent",
            "name": "Test Agent",
            "description": "A test agent for registry testing",
            "host": "localhost",
            "port": 8000,
            "capabilities": ["math_query", "math_computation"],
            "status": "active"
        }
        
        registry.register_agent("test_agent", agent_info)
        logger.info(f"Registered test agent: {agent_info['name']}")
        
        # Register another test agent
        agent_info2 = {
            "id": "test_agent2",
            "name": "Test Agent 2",
            "description": "Another test agent for registry testing",
            "host": "localhost",
            "port": 8001,
            "capabilities": ["math_query", "visualization"],
            "status": "active"
        }
        
        registry.register_agent("test_agent2", agent_info2)
        logger.info(f"Registered test agent: {agent_info2['name']}")
        
        # Find agent by capability
        math_agents = registry.find_agent_by_capability("math_query")
        if not math_agents or len(math_agents) != 2:
            logger.error(f"Expected 2 agents with math_query capability, found {len(math_agents)}")
            return False
        
        logger.info(f"Found {len(math_agents)} agents with math_query capability")
        
        # Find agent by ID
        agent = registry.get_agent_info("test_agent")
        if not agent or agent["name"] != "Test Agent":
            logger.error("Failed to get agent by ID")
            return False
        
        logger.info(f"Retrieved agent by ID: {agent['name']}")
        
        # Find agents by multiple capabilities
        computation_agents = registry.find_agent_by_capability("math_computation")
        if not computation_agents or len(computation_agents) != 1:
            logger.error(f"Expected 1 agent with math_computation capability, found {len(computation_agents)}")
            return False
        
        logger.info(f"Found {len(computation_agents)} agents with math_computation capability")
        
        # Update agent status
        registry.update_agent_status("test_agent", "busy")
        agent = registry.get_agent_info("test_agent")
        if not agent or agent["status"] != "busy":
            logger.error(f"Failed to update agent status, expected 'busy', got '{agent.get('status')}'")
            return False
        
        logger.info(f"Updated agent status to: {agent['status']}")
        
        # Deregister agent
        registry.deregister_agent("test_agent")
        agent = registry.get_agent_info("test_agent")
        if agent:
            logger.error("Failed to deregister agent")
            return False
        
        logger.info("Successfully deregistered test agent")
        
        logger.info("Agent registry tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Agent registry test failed: {e}")
        return False

def test_workflow_orchestration() -> bool:
    """
    Test workflow orchestration.
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing workflow orchestration...")
    
    try:
        # Initialize agent registry
        registry = AgentRegistry()
        
        # Register test agents
        registry.register_agent("llm_agent", {
            "id": "llm_agent",
            "name": "LLM Agent",
            "capabilities": ["math_query", "query_classification"],
            "status": "active"
        })
        
        registry.register_agent("math_agent", {
            "id": "math_agent",
            "name": "Math Computation Agent",
            "capabilities": ["math_computation"],
            "status": "active"
        })
        
        registry.register_agent("viz_agent", {
            "id": "viz_agent",
            "name": "Visualization Agent",
            "capabilities": ["visualization"],
            "status": "active"
        })
        
        # Define mock message bus for testing
        class MockMessageBus:
            def __init__(self):
                self.sent_messages = []
                self.callbacks = {}
            
            def send_message(self, recipient, message, callback=None):
                self.sent_messages.append((recipient, message))
                if callback:
                    self.callbacks[recipient] = callback
                return True
            
            def simulate_response(self, recipient, response_message):
                if recipient in self.callbacks:
                    self.callbacks[recipient](response_message)
        
        message_bus = MockMessageBus()
        
        # Define test workflow
        math_problem_steps = [
            WorkflowStep(
                id="classify_query",
                required_capability="query_classification",
                message_type="math_query",
                input_keys=["query"],
                output_keys=["domain", "expression"]
            ),
            WorkflowStep(
                id="compute_result",
                required_capability="math_computation",
                message_type="math_computation",
                input_keys=["expression", "domain"],
                output_keys=["result", "steps"]
            ),
            WorkflowStep(
                id="generate_visualization",
                required_capability="visualization",
                message_type="visualization_request",
                input_keys=["expression", "result"],
                output_keys=["visualization"]
            )
        ]
        
        math_problem_workflow = WorkflowDefinition(
            id="math_problem_solving",
            name="Mathematical Problem Solving",
            description="Workflow for solving mathematical problems",
            steps=math_problem_steps
        )
        
        workflow_definitions = {
            "math_problem_solving": math_problem_workflow
        }
        
        # Initialize orchestration manager
        orchestration_manager = OrchestrationManager(
            message_bus=message_bus,
            agent_registry=registry,
            workflow_definitions=workflow_definitions
        )
        
        # Start a test workflow
        initial_data = {
            "query": "Solve the equation: 3x + 2 = 8"
        }
        
        workflow_id = orchestration_manager.start_workflow(
            workflow_type="math_problem_solving",
            initial_data=initial_data
        )
        
        logger.info(f"Started test workflow with ID: {workflow_id}")
        
        # Check if a message was sent to the first agent
        if not message_bus.sent_messages:
            logger.error("No messages sent after starting workflow")
            return False
        
        recipient, message = message_bus.sent_messages[0]
        if recipient != "llm_agent":
            logger.error(f"Expected message to be sent to llm_agent, got {recipient}")
            return False
        
        logger.info(f"Message sent to correct agent: {recipient}")
        
        # Simulate response from first agent
        message_data = json.loads(message)
        response = {
            "header": {
                "message_id": "resp_" + message_data["header"]["message_id"],
                "message_type": "math_response",
                "sender": "llm_agent",
                "recipient": "orchestration_manager",
                "timestamp": int(time.time() * 1000),
                "priority": "normal",
                "correlation_id": message_data["header"]["message_id"]
            },
            "body": {
                "success": True,
                "response_content": {
                    "output_data": {
                        "domain": "algebra",
                        "expression": "3x + 2 = 8"
                    }
                }
            }
        }
        
        message_bus.simulate_response("llm_agent", json.dumps(response))
        
        # Check if a message was sent to the second agent
        if len(message_bus.sent_messages) < 2:
            logger.error("No message sent to second agent after first response")
            return False
        
        recipient, message = message_bus.sent_messages[1]
        if recipient != "math_agent":
            logger.error(f"Expected message to be sent to math_agent, got {recipient}")
            return False
        
        logger.info(f"Message sent to correct agent: {recipient}")
        
        # Simulate response from second agent
        message_data = json.loads(message)
        response = {
            "header": {
                "message_id": "resp_" + message_data["header"]["message_id"],
                "message_type": "computation_result",
                "sender": "math_agent",
                "recipient": "orchestration_manager",
                "timestamp": int(time.time() * 1000),
                "priority": "normal",
                "correlation_id": message_data["header"]["message_id"]
            },
            "body": {
                "success": True,
                "response_content": {
                    "output_data": {
                        "result": "x = 2",
                        "steps": ["3x + 2 = 8", "3x = 6", "x = 2"]
                    }
                }
            }
        }
        
        message_bus.simulate_response("math_agent", json.dumps(response))
        
        # Check if a message was sent to the third agent
        if len(message_bus.sent_messages) < 3:
            logger.error("No message sent to third agent after second response")
            return False
        
        recipient, message = message_bus.sent_messages[2]
        if recipient != "viz_agent":
            logger.error(f"Expected message to be sent to viz_agent, got {recipient}")
            return False
        
        logger.info(f"Message sent to correct agent: {recipient}")
        
        # Simulate response from third agent
        message_data = json.loads(message)
        response = {
            "header": {
                "message_id": "resp_" + message_data["header"]["message_id"],
                "message_type": "visualization_result",
                "sender": "viz_agent",
                "recipient": "orchestration_manager",
                "timestamp": int(time.time() * 1000),
                "priority": "normal",
                "correlation_id": message_data["header"]["message_id"]
            },
            "body": {
                "success": True,
                "response_content": {
                    "output_data": {
                        "visualization": "http://example.com/plot.png"
                    }
                }
            }
        }
        
        message_bus.simulate_response("viz_agent", json.dumps(response))
        
        # Check workflow status
        workflow = orchestration_manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Failed to get workflow with ID: {workflow_id}")
            return False
        
        if workflow["state"] != "completed":
            logger.error(f"Expected workflow state to be 'completed', got '{workflow['state']}'")
            return False
        
        logger.info(f"Workflow completed with state: {workflow['state']}")
        
        # Check workflow data
        expected_keys = ["query", "domain", "expression", "result", "steps", "visualization"]
        for key in expected_keys:
            if key not in workflow["data"]:
                logger.error(f"Expected key '{key}' in workflow data")
                return False
        
        logger.info(f"Workflow data contains all expected keys: {', '.join(expected_keys)}")
        
        logger.info("Workflow orchestration tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Workflow orchestration test failed: {e}")
        return False

def main():
    """Main function for integration testing."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # Run requested tests
    if args.test_mode in ['basic', 'model', 'all']:
        logger.info("Running Core LLM Agent tests...")
        results['core_llm_agent'] = test_core_llm_agent(
            cache_dir=args.cache_dir,
            quantization=args.quantization,
            use_vllm=args.use_vllm
        )
    
    if args.test_mode in ['db', 'all']:
        logger.info("Running database connection tests...")
        results['database_connections'] = test_database_connections(
            mongodb_uri=args.mongodb_uri
        )
    
    if args.test_mode in ['message', 'all']:
        logger.info("Running message bus tests...")
        results['message_bus'] = test_message_bus(
            rabbitmq_host=args.rabbitmq_host,
            rabbitmq_port=args.rabbitmq_port
        )
    
    if args.test_mode in ['agent', 'all']:
        logger.info("Running agent registry tests...")
        results['agent_registry'] = test_agent_registry()
    
    if args.test_mode in ['workflow', 'all']:
        logger.info("Running workflow orchestration tests...")
        results['workflow_orchestration'] = test_workflow_orchestration()
    
    # Calculate overall result
    if results:
        overall_result = all(results.values())
        results['overall'] = overall_result
        
        logger.info(f"Overall test result: {'PASS' if overall_result else 'FAIL'}")
        
        # Save results
        results_path = os.path.join(args.output_dir, "integration_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to: {results_path}")
        
        # Exit with appropriate status code
        sys.exit(0 if overall_result else 1)
    else:
        logger.warning("No tests were run")
        sys.exit(1)

if __name__ == "__main__":
    main()

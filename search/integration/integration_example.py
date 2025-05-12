"""
Integration example for the Search Agent.

This module provides an example of how to integrate the Search Agent with
the Core LLM Agent in the Mathematical Multimodal LLM System.
"""

import logging
import os
from typing import Dict, Any

from search.agent.search_agent import SearchAgent
from search.integration.llm_connector import LLMConnector
from core.agent.llm_agent import CoreLLMAgent  # Import from your existing project
from orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus  # Import from your existing project

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_search_agent() -> SearchAgent:
    """
    Set up and configure the Search Agent.
    
    Returns:
        Configured SearchAgent instance
    """
    # Get API keys from environment
    config = {
        "google_api_key": os.environ.get("GOOGLE_API_KEY", ""),
        "google_cx": os.environ.get("GOOGLE_CX", ""),
        "wolfram_app_id": os.environ.get("WOLFRAM_APP_ID", ""),
        "message_bus_host": os.environ.get("RABBITMQ_HOST", "localhost"),
        "message_bus_port": int(os.environ.get("RABBITMQ_PORT", "5672"))
    }
    
    # Create and return SearchAgent
    return SearchAgent(config=config)

def setup_llm_connector(search_agent: SearchAgent) -> LLMConnector:
    """
    Set up the LLM Connector.
    
    Args:
        search_agent: SearchAgent instance
        
    Returns:
        Configured LLMConnector instance
    """
    return LLMConnector(search_agent=search_agent)

def register_with_orchestration(message_bus: RabbitMQBus, search_agent: SearchAgent):
    """
    Register Search Agent with the Orchestration Manager.
    
    Args:
        message_bus: Message bus for communication
        search_agent: SearchAgent instance
    """
    # Define capabilities
    capabilities = [
        "search_internet",
        "verify_information",
        "retrieve_mathematical_content",
        "supplement_knowledge"
    ]
    
    # Create registration message
    registration_message = {
        "agent_id": "search_agent",
        "agent_type": "search",
        "capabilities": capabilities,
        "status": "active"
    }
    
    # Send registration message
    message_bus.send_message(
        recipient="orchestration_manager",
        message_body=registration_message,
        message_type="agent_registration"
    )
    
    logger.info("Search Agent registered with Orchestration Manager")

def enhance_llm_response(core_llm: CoreLLMAgent, llm_connector: LLMConnector, query: str, domain: str = None) -> Dict[str, Any]:
    """
    Enhance LLM response with search results.
    
    Args:
        core_llm: CoreLLMAgent instance
        llm_connector: LLMConnector instance
        query: User query
        domain: Optional mathematical domain
        
    Returns:
        Enhanced response information
    """
    # Enhance prompt with search
    enhanced_prompt_info = llm_connector.enhance_prompt(query, domain)
    
    # Generate LLM response with enhanced prompt
    llm_response = core_llm.generate_response(enhanced_prompt_info["enhanced_prompt"])
    
    # Enhance response with citations if search was performed
    if enhanced_prompt_info["context_added"]:
        enhanced_response = llm_connector.enhance_response(
            llm_response, 
            enhanced_prompt_info["search_results"],
            add_citations=True,
            citation_style="standard"
        )
    else:
        enhanced_response = llm_response
    
    # Return full information
    return {
        "original_query": query,
        "enhanced_prompt": enhanced_prompt_info["enhanced_prompt"] if enhanced_prompt_info["context_added"] else None,
        "original_response": llm_response,
        "enhanced_response": enhanced_response,
        "search_results": enhanced_prompt_info["search_results"],
        "search_performed": enhanced_prompt_info["context_added"]
    }

def run_example():
    """Run an integration example."""
    # Set up components
    search_agent = setup_search_agent()
    llm_connector = setup_llm_connector(search_agent)
    
    # Set up message bus
    message_bus = RabbitMQBus(
        host=os.environ.get("RABBITMQ_HOST", "localhost"),
        port=int(os.environ.get("RABBITMQ_PORT", "5672"))
    )
    
    # Register with orchestration
    register_with_orchestration(message_bus, search_agent)
    
    # Create Core LLM Agent
    core_llm = CoreLLMAgent()  # Your existing Core LLM Agent
    
    # Example queries
    queries = [
        {
            "query": "What is the Riemann Hypothesis?",
            "domain": "number_theory"
        },
        {
            "query": "Explain the Fundamental Theorem of Calculus",
            "domain": "calculus"
        },
        {
            "query": "What are eigenvalues and eigenvectors?",
            "domain": "linear_algebra"
        }
    ]
    
    # Process each query
    for query_info in queries:
        query = query_info["query"]
        domain = query_info["domain"]
        
        logger.info(f"Processing query: {query}")
        
        # Enhance LLM response
        result = enhance_llm_response(core_llm, llm_connector, query, domain)
        
        # Log results
        if result["search_performed"]:
            logger.info("Search was performed and incorporated into response")
        else:
            logger.info("No search was performed for this query")
        
        logger.info(f"Enhanced response: {result['enhanced_response'][:100]}...")
        logger.info("-" * 50)

if __name__ == "__main__":
    # Run the example
    run_example()
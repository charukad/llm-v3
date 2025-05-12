"""
RabbitMQ configuration for the Mathematical Multimodal LLM System.

This module provides configuration settings for RabbitMQ in different environments
and utility functions for connection management.
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "development": {
        "host": "localhost",
        "port": 5672,
        "virtual_host": "/",
        "username": "guest",
        "password": "guest",
        "connection_attempts": 3,
        "retry_delay": 5,
        "heartbeat": 600,
        "blocked_connection_timeout": 300
    },
    "staging": {
        "host": "localhost",
        "port": 5672,
        "virtual_host": "/",
        "username": "math_llm",
        "password": "password",  # Should be overridden by environment variables
        "connection_attempts": 5,
        "retry_delay": 5,
        "heartbeat": 600,
        "blocked_connection_timeout": 300
    },
    "production": {
        "host": "localhost",
        "port": 5672,
        "virtual_host": "/",
        "username": "math_llm",
        "password": "password",  # Should be overridden by environment variables
        "connection_attempts": 10,
        "retry_delay": 5,
        "heartbeat": 600,
        "blocked_connection_timeout": 300
    }
}

# Exchange and queue definitions
EXCHANGES = {
    "main": {
        "name": "math_llm.main",
        "type": "topic",
        "durable": True,
        "auto_delete": False
    },
    "events": {
        "name": "math_llm.events",
        "type": "fanout",
        "durable": True,
        "auto_delete": False
    },
    "dlx": {
        "name": "math_llm.dlx",
        "type": "direct",
        "durable": True,
        "auto_delete": False
    }
}

QUEUES = {
    "math_query": {
        "name": "math_llm.math_query",
        "durable": True,
        "exclusive": False,
        "auto_delete": False,
        "arguments": {
            "x-dead-letter-exchange": "math_llm.dlx",
            "x-dead-letter-routing-key": "math_llm.dead_letters",
            "x-message-ttl": 60000  # 1 minute
        }
    },
    "math_computation": {
        "name": "math_llm.math_computation",
        "durable": True,
        "exclusive": False,
        "auto_delete": False,
        "arguments": {
            "x-dead-letter-exchange": "math_llm.dlx",
            "x-dead-letter-routing-key": "math_llm.dead_letters",
            "x-message-ttl": 60000  # 1 minute
        }
    },
    "handwriting_recognition": {
        "name": "math_llm.handwriting_recognition",
        "durable": True,
        "exclusive": False,
        "auto_delete": False,
        "arguments": {
            "x-dead-letter-exchange": "math_llm.dlx",
            "x-dead-letter-routing-key": "math_llm.dead_letters",
            "x-message-ttl": 120000  # 2 minutes
        }
    },
    "visualization": {
        "name": "math_llm.visualization",
        "durable": True,
        "exclusive": False,
        "auto_delete": False,
        "arguments": {
            "x-dead-letter-exchange": "math_llm.dlx",
            "x-dead-letter-routing-key": "math_llm.dead_letters",
            "x-message-ttl": 60000  # 1 minute
        }
    },
    "search": {
        "name": "math_llm.search",
        "durable": True,
        "exclusive": False,
        "auto_delete": False,
        "arguments": {
            "x-dead-letter-exchange": "math_llm.dlx",
            "x-dead-letter-routing-key": "math_llm.dead_letters",
            "x-message-ttl": 60000  # 1 minute
        }
    },
    "events": {
        "name": "math_llm.events",
        "durable": True,
        "exclusive": False,
        "auto_delete": False,
        "arguments": {}
    },
    "dead_letters": {
        "name": "math_llm.dead_letters",
        "durable": True,
        "exclusive": False,
        "auto_delete": False,
        "arguments": {}
    }
}

# Routing keys
ROUTING_KEYS = {
    "math_query": "math.query.*",
    "math_computation": "math.computation.*",
    "handwriting_recognition": "math.handwriting.*",
    "visualization": "math.visualization.*",
    "search": "math.search.*",
    "events": "events.*"
}

def get_rabbitmq_config(environment: str = None) -> Dict[str, Any]:
    """
    Get RabbitMQ configuration for the specified environment.
    
    Args:
        environment: Environment name (development, staging, production)
        
    Returns:
        RabbitMQ configuration dictionary
    """
    # Determine environment
    if environment is None:
        environment = os.environ.get("APP_ENV", "development")
    
    # Validate environment
    if environment not in DEFAULT_CONFIG:
        logger.warning(f"Unknown environment: {environment}, using development")
        environment = "development"
    
    # Get base configuration
    config = DEFAULT_CONFIG[environment].copy()
    
    # Override with environment variables if available
    if os.environ.get("RABBITMQ_HOST"):
        config["host"] = os.environ.get("RABBITMQ_HOST")
    
    if os.environ.get("RABBITMQ_PORT"):
        config["port"] = int(os.environ.get("RABBITMQ_PORT"))
    
    if os.environ.get("RABBITMQ_VIRTUAL_HOST"):
        config["virtual_host"] = os.environ.get("RABBITMQ_VIRTUAL_HOST")
    
    if os.environ.get("RABBITMQ_USERNAME"):
        config["username"] = os.environ.get("RABBITMQ_USERNAME")
    
    if os.environ.get("RABBITMQ_PASSWORD"):
        config["password"] = os.environ.get("RABBITMQ_PASSWORD")
    
    logger.info(f"Using RabbitMQ configuration for environment: {environment}")
    return config

def get_exchange_config(exchange_key: str) -> Dict[str, Any]:
    """
    Get configuration for an exchange.
    
    Args:
        exchange_key: Exchange key
        
    Returns:
        Exchange configuration
    """
    # Check if the exchange key exists
    if exchange_key not in EXCHANGES:
        logger.warning(f"Unknown exchange key: {exchange_key}, using default")
        exchange_key = "main"
    
    # Get exchange configuration
    exchange_config = EXCHANGES[exchange_key].copy()
    
    # Override name with environment variable if available
    env_var = f"RABBITMQ_EXCHANGE_{exchange_key.upper()}"
    if os.environ.get(env_var):
        exchange_config["name"] = os.environ.get(env_var)
    
    return exchange_config

def get_queue_config(queue_key: str) -> Dict[str, Any]:
    """
    Get configuration for a queue.
    
    Args:
        queue_key: Queue key
        
    Returns:
        Queue configuration
    """
    # Check if the queue key exists
    if queue_key not in QUEUES:
        logger.warning(f"Unknown queue key: {queue_key}, using math_query")
        queue_key = "math_query"
    
    # Get queue configuration
    queue_config = QUEUES[queue_key].copy()
    
    # Override name with environment variable if available
    env_var = f"RABBITMQ_QUEUE_{queue_key.upper()}"
    if os.environ.get(env_var):
        queue_config["name"] = os.environ.get(env_var)
    
    # If we override the DLX name, update the arguments
    if queue_config["arguments"].get("x-dead-letter-exchange"):
        dlx_config = get_exchange_config("dlx")
        queue_config["arguments"]["x-dead-letter-exchange"] = dlx_config["name"]
    
    return queue_config

def get_routing_key(routing_key: str) -> str:
    """
    Get a routing key.
    
    Args:
        routing_key: Routing key name
        
    Returns:
        Routing key pattern
    """
    # Check if the routing key exists
    if routing_key not in ROUTING_KEYS:
        logger.warning(f"Unknown routing key: {routing_key}, using default")
        routing_key = "math_query"
    
    # Get routing key pattern
    return ROUTING_KEYS[routing_key]

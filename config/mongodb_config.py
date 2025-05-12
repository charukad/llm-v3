"""
MongoDB configuration for the Mathematical Multimodal LLM System.

This module provides configuration settings for MongoDB in different environments
and utility functions for connection management.
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "development": {
        "uri": "mongodb://localhost:27017",
        "database": "math_llm_system_dev",
        "options": {
            "maxPoolSize": 10,
            "connectTimeoutMS": 30000,
            "socketTimeoutMS": 45000,
            "serverSelectionTimeoutMS": 30000,
            "w": 1,  # Write concern: 1 = ack from primary
            "retryWrites": True,
            "retryReads": True
        }
    },
    "staging": {
        "uri": "mongodb://localhost:27017",
        "database": "math_llm_system_staging",
        "options": {
            "maxPoolSize": 50,
            "connectTimeoutMS": 30000,
            "socketTimeoutMS": 45000,
            "serverSelectionTimeoutMS": 30000,
            "w": 1,
            "retryWrites": True,
            "retryReads": True
        }
    },
    "production": {
        "uri": "mongodb://localhost:27017",
        "database": "math_llm_system_prod",
        "options": {
            "maxPoolSize": 100,
            "connectTimeoutMS": 30000,
            "socketTimeoutMS": 45000,
            "serverSelectionTimeoutMS": 30000,
            "w": "majority",  # Write concern: majority = ack from majority of replica set
            "retryWrites": True,
            "retryReads": True,
            "readPreference": "secondaryPreferred"  # Read from secondaries when possible
        }
    }
}

# Collection names
COLLECTIONS = {
    "users": "users",
    "conversations": "conversations",
    "interactions": "interactions",
    "expressions": "math_expressions",
    "handwritten_inputs": "handwritten_inputs",
    "visualizations": "visualizations",
    "knowledge": "math_knowledge",
    "models": "models",
    "model_versions": "model_versions",
    "model_configs": "model_configs",
    "model_metrics": "model_metrics",
    "workflows": "workflows",
    "agents": "agent_registry",
    "stats": "system_stats",
    "logs": "system_logs"
}

def get_mongo_config(environment: str = None) -> Dict[str, Any]:
    """
    Get MongoDB configuration for the specified environment.
    
    Args:
        environment: Environment name (development, staging, production)
        
    Returns:
        MongoDB configuration dictionary
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
    if os.environ.get("MONGODB_URI"):
        config["uri"] = os.environ.get("MONGODB_URI")
    
    if os.environ.get("MONGODB_DATABASE"):
        config["database"] = os.environ.get("MONGODB_DATABASE")
    
    logger.info(f"Using MongoDB configuration for environment: {environment}")
    return config

def get_collection_name(collection_key: str) -> str:
    """
    Get the actual collection name for a logical collection key.
    
    Args:
        collection_key: Logical collection key
        
    Returns:
        Actual collection name
    """
    # Check if the collection key exists
    if collection_key not in COLLECTIONS:
        logger.warning(f"Unknown collection key: {collection_key}, using as-is")
        return collection_key
    
    # Get collection name
    collection_name = COLLECTIONS[collection_key]
    
    # Override with environment variable if available
    env_var = f"MONGODB_COLLECTION_{collection_key.upper()}"
    if os.environ.get(env_var):
        collection_name = os.environ.get(env_var)
    
    return collection_name

"""
Test script for the database layer of the Mathematical Multimodal LLM System.

This script tests the basic functionality of the database repositories.
"""

import logging
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mongodb_wrapper():
    """Test the MongoDB wrapper."""
    logger.info("Testing MongoDB wrapper...")
    
    try:
        from database.access.mongodb_wrapper import MongoDBWrapper
        
        # Connect to MongoDB
        mongodb = MongoDBWrapper(
            connection_string="mongodb://localhost:27017/",
            database_name="math_llm_test"
        )
        
        # Test basic operations
        test_collection = mongodb.get_collection("test_collection")
        
        # Insert a document
        doc_id = mongodb.insert_one("test_collection", {"test": "value", "timestamp": datetime.now()})
        logger.info(f"Inserted document: {doc_id}")
        
        # Find the document
        doc = mongodb.find_one("test_collection", {"_id": ObjectId(doc_id)})
        logger.info(f"Found document: {doc}")
        
        # Update the document
        mongodb.update_one("test_collection", {"_id": ObjectId(doc_id)}, {"$set": {"updated": True}})
        
        # Verify update
        updated_doc = mongodb.find_one("test_collection", {"_id": ObjectId(doc_id)})
        logger.info(f"Updated document: {updated_doc}")
        
        # Clean up
        test_collection.drop()
        
        logger.info("MongoDB wrapper test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"MongoDB wrapper test failed: {e}")
        return False

def test_conversation_repository():
    """Test the conversation repository."""
    logger.info("Testing conversation repository...")
    
    try:
        from database.access.mongodb_wrapper import MongoDBWrapper
        from database.access.conversation_repository import ConversationRepository
        
        # Initialize repositories
        mongodb = MongoDBWrapper(
            connection_string="mongodb://localhost:27017/",
            database_name="math_llm_test"
        )
        repo = ConversationRepository(mongodb)
        
        # Create a test user ID
        test_user_id = "test_user_" + datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create a conversation
        conversation_id = repo.create_conversation(
            user_id=test_user_id,
            title="Test Conversation"
        )
        logger.info(f"Created conversation: {conversation_id}")
        
        # Add an interaction
        interaction_id = repo.add_interaction(
            conversation_id=conversation_id,
            user_input={"text": "What is the derivative of x^2?", "math_domains": ["calculus"]},
            system_response={"text": "The derivative of x^2 is 2x", "latex": "\\frac{d}{dx}x^2 = 2x"}
        )
        logger.info(f"Added interaction: {interaction_id}")
        
        # Get conversation
        conversation = repo.get_conversation(conversation_id)
        logger.info(f"Retrieved conversation: {conversation}")
        
        # Get interactions
        interactions = repo.get_interactions(conversation_id)
        logger.info(f"Retrieved {len(interactions)} interactions")
        
        # Get user conversations
        conversations = repo.get_user_conversations(test_user_id)
        logger.info(f"Retrieved {len(conversations)} conversations for user")
        
        # Clean up
        mongodb.get_collection("conversations").drop()
        mongodb.get_collection("interactions").drop()
        
        logger.info("Conversation repository test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Conversation repository test failed: {e}")
        return False

def test_expression_repository():
    """Test the expression repository."""
    logger.info("Testing expression repository...")
    
    try:
        from database.access.mongodb_wrapper import MongoDBWrapper
        from database.access.conversation_repository import ConversationRepository
        from database.access.expression_repository import ExpressionRepository
        from bson.objectid import ObjectId
        
        # Initialize repositories
        mongodb = MongoDBWrapper(
            connection_string="mongodb://localhost:27017/",
            database_name="math_llm_test"
        )
        conv_repo = ConversationRepository(mongodb)
        expr_repo = ExpressionRepository(mongodb)
        
        # Create a test conversation and interaction
        test_user_id = "test_user_" + datetime.now().strftime("%Y%m%d%H%M%S")
        conversation_id = conv_repo.create_conversation(test_user_id)
        interaction_id = conv_repo.add_interaction(
            conversation_id=conversation_id,
            user_input={"text": "Solve x^2 - 5x + 6 = 0"},
            system_response={"text": "The solutions are x = 2 and x = 3"}
        )
        
        # Store an expression
        expression_id = expr_repo.store_expression(
            interaction_id=interaction_id,
            latex_representation="x^2 - 5x + 6 = 0",
            symbolic_representation="Eq(x**2 - 5*x + 6, 0)",
            math_domain="algebra",
            metadata={"type": "equation", "solutions": ["2", "3"]}
        )
        logger.info(f"Stored expression: {expression_id}")
        
        # Get the expression
        expression = expr_repo.get_expression(expression_id)
        logger.info(f"Retrieved expression: {expression}")
        
        # Find similar expressions
        similar = expr_repo.find_similar_expressions("x^2")
        logger.info(f"Found {len(similar)} similar expressions")
        
        # Get expressions by domain
        by_domain = expr_repo.get_expressions_by_domain("algebra")
        logger.info(f"Found {len(by_domain)} expressions in algebra domain")
        
        # Clean up
        mongodb.get_collection("mathematical_expressions").drop()
        mongodb.get_collection("interactions").drop()
        mongodb.get_collection("conversations").drop()
        
        logger.info("Expression repository test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Expression repository test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting database tests...")
    
    # Import ObjectId here to avoid circular imports in tests
    from bson.objectid import ObjectId
    
    # Ensure MongoDB is running
    from pymongo import MongoClient
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        logger.info("MongoDB server is running")
    except Exception as e:
        logger.error(f"MongoDB server is not available: {e}")
        logger.error("Please ensure MongoDB is running before running tests")
        sys.exit(1)
    
    # Run tests
    wrapper_success = test_mongodb_wrapper()
    logger.info("-----------------------------------")
    
    conversation_success = test_conversation_repository()
    logger.info("-----------------------------------")
    
    expression_success = test_expression_repository()
    
    # Report results
    logger.info("===================================")
    logger.info("Test Results:")
    logger.info(f"MongoDB Wrapper: {'PASSED' if wrapper_success else 'FAILED'}")
    logger.info(f"Conversation Repository: {'PASSED' if conversation_success else 'FAILED'}")
    logger.info(f"Expression Repository: {'PASSED' if expression_success else 'FAILED'}")
    
    if wrapper_success and conversation_success and expression_success:
        logger.info("All database tests completed successfully")
        sys.exit(0)
    else:
        logger.error("One or more database tests failed")
        sys.exit(1)

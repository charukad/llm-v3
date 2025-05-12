"""
Conversation repository for storing and retrieving conversations and interactions.

This module provides a repository for managing conversation data in MongoDB.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from bson.objectid import ObjectId
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

class ConversationRepository:
    """Repository for managing conversations and interactions."""
    
    def __init__(self, mongodb_wrapper):
        """
        Initialize the conversation repository.
        
        Args:
            mongodb_wrapper: MongoDB wrapper instance
        """
        self.mongodb = mongodb_wrapper
        self.conversations = self.mongodb.get_collection("conversations")
        self.interactions = self.mongodb.get_collection("interactions")
    
    def create_conversation(self, user_id: str, title: Optional[str] = None) -> Optional[str]:
        """
        Create a new conversation.
        
        Args:
            user_id: ID of the user
            title: Optional title for the conversation
            
        Returns:
            ID of the created conversation or None if failed
        """
        try:
            # Create conversation document
            conversation = {
                "user_id": user_id,
                "title": title or "New Conversation",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "interaction_count": 0,
                "math_domains": [],
                "status": "active"
            }
            
            # Insert into database
            result = self.conversations.insert_one(conversation)
            conversation_id = str(result.inserted_id)
            
            logger.info(f"Created conversation {conversation_id} for user {user_id}")
            return conversation_id
            
        except PyMongoError as e:
            logger.error(f"Failed to create conversation: {e}")
            return None
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation document or None if not found
        """
        try:
            # Convert string ID to ObjectId
            obj_id = ObjectId(conversation_id)
            
            # Retrieve from database
            conversation = self.conversations.find_one({"_id": obj_id})
            
            if conversation:
                # Convert ObjectId to string for serialization
                conversation["_id"] = str(conversation["_id"])
                return conversation
            else:
                logger.warning(f"Conversation {conversation_id} not found")
                return None
                
        except PyMongoError as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
    
    def add_interaction(self, conversation_id: str, 
                       user_input: Dict[str, Any], 
                       system_response: Dict[str, Any]) -> Optional[str]:
        """
        Add an interaction to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            user_input: User input data
            system_response: System response data
            
        Returns:
            ID of the created interaction or None if failed
        """
        try:
            # Convert string ID to ObjectId
            conv_obj_id = ObjectId(conversation_id)
            
            # Check if conversation exists
            if not self.conversations.find_one({"_id": conv_obj_id}):
                logger.error(f"Cannot add interaction: Conversation {conversation_id} not found")
                return None
            
            # Create interaction document
            interaction = {
                "conversation_id": conv_obj_id,
                "timestamp": datetime.now(),
                "user_input": user_input,
                "system_response": system_response,
                "math_expressions": [],  # Will be populated by references
                "visualizations": []     # Will be populated by references
            }
            
            # Insert into database
            result = self.interactions.insert_one(interaction)
            interaction_id = str(result.inserted_id)
            
            # Update conversation metadata
            self.conversations.update_one(
                {"_id": conv_obj_id},
                {
                    "$inc": {"interaction_count": 1},
                    "$set": {"updated_at": datetime.now()},
                    "$addToSet": {"math_domains": {"$each": user_input.get("math_domains", [])}}
                }
            )
            
            logger.info(f"Added interaction {interaction_id} to conversation {conversation_id}")
            return interaction_id
            
        except PyMongoError as e:
            logger.error(f"Failed to add interaction to conversation {conversation_id}: {e}")
            return None
    
    def get_interactions(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get interactions for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of interaction documents
        """
        try:
            # Convert string ID to ObjectId
            conv_obj_id = ObjectId(conversation_id)
            
            # Retrieve from database
            cursor = self.interactions.find(
                {"conversation_id": conv_obj_id}
            ).sort("timestamp", -1).limit(limit)
            
            # Convert ObjectIds to strings for serialization
            interactions = []
            for interaction in cursor:
                interaction["_id"] = str(interaction["_id"])
                interaction["conversation_id"] = str(interaction["conversation_id"])
                interactions.append(interaction)
            
            return interactions
            
        except PyMongoError as e:
            logger.error(f"Failed to get interactions for conversation {conversation_id}: {e}")
            return []
    
    def get_user_conversations(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get conversations for a user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation documents
        """
        try:
            # Retrieve from database
            cursor = self.conversations.find(
                {"user_id": user_id}
            ).sort("updated_at", -1).limit(limit)
            
            # Convert ObjectIds to strings for serialization
            conversations = []
            for conversation in cursor:
                conversation["_id"] = str(conversation["_id"])
                conversations.append(conversation)
            
            return conversations
            
        except PyMongoError as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}")
            return []

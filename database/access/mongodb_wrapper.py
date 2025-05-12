"""
MongoDB connection and utility functions.

This module provides a wrapper for MongoDB connections and basic operations.
"""

import logging
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

class MongoDBWrapper:
    """Wrapper for MongoDB connection and operations."""
    
    def __init__(self, connection_string: str, database_name: str = "math_llm_system"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database to use
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        
        self._connect()
        
    def _connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            logger.info(f"Connecting to MongoDB: {self.connection_string}")
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            # Ping database to check connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a reference to a MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB collection reference
        """
        return self.db[collection_name]
    
    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document into a collection.
        
        Args:
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            ID of the inserted document
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert document into {collection_name}: {e}")
            raise
    
    def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query to find the document
            
        Returns:
            Found document or None
        """
        try:
            collection = self.get_collection(collection_name)
            return collection.find_one(query)
        except Exception as e:
            logger.error(f"Failed to find document in {collection_name}: {e}")
            raise
    
    def find_many(self, collection_name: str, query: Dict[str, Any], 
                 limit: int = 0, sort: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Find multiple documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query to find the documents
            limit: Maximum number of documents to return (0 for no limit)
            sort: Sort specification
            
        Returns:
            List of found documents
        """
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query)
            
            if limit > 0:
                cursor = cursor.limit(limit)
                
            if sort:
                cursor = cursor.sort(sort)
                
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to find documents in {collection_name}: {e}")
            raise
    
    def update_one(self, collection_name: str, query: Dict[str, Any], 
                  update: Dict[str, Any], upsert: bool = False) -> int:
        """
        Update a single document in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query to find the document
            update: Update operations
            upsert: Whether to insert if document doesn't exist
            
        Returns:
            Number of modified documents
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(query, update, upsert=upsert)
            return result.modified_count
        except Exception as e:
            logger.error(f"Failed to update document in {collection_name}: {e}")
            raise
    
    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")

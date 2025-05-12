"""
Mathematical expression repository for storing and retrieving mathematical expressions.

This module provides a repository for managing mathematical expression data in MongoDB.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from bson.objectid import ObjectId
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

class ExpressionRepository:
    """Repository for managing mathematical expressions."""
    
    def __init__(self, mongodb_wrapper):
        """
        Initialize the expression repository.
        
        Args:
            mongodb_wrapper: MongoDB wrapper instance
        """
        self.mongodb = mongodb_wrapper
        self.expressions = self.mongodb.get_collection("mathematical_expressions")
        self.interactions = self.mongodb.get_collection("interactions")
    
    def store_expression(self, interaction_id: str, 
                        latex_representation: str,
                        symbolic_representation: Optional[str] = None,
                        math_domain: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Store a mathematical expression.
        
        Args:
            interaction_id: ID of the interaction
            latex_representation: LaTeX representation of the expression
            symbolic_representation: Symbolic representation (e.g., SymPy)
            math_domain: Mathematical domain (algebra, calculus, etc.)
            metadata: Additional metadata
            
        Returns:
            ID of the stored expression or None if failed
        """
        try:
            # Convert string ID to ObjectId
            interaction_obj_id = ObjectId(interaction_id)
            
            # Check if interaction exists
            if not self.interactions.find_one({"_id": interaction_obj_id}):
                logger.error(f"Cannot store expression: Interaction {interaction_id} not found")
                return None
            
            # Create expression document
            expression = {
                "interaction_id": interaction_obj_id,
                "latex_representation": latex_representation,
                "symbolic_representation": symbolic_representation,
                "math_domain": math_domain,
                "metadata": metadata or {},
                "created_at": datetime.now()
            }
            
            # Insert into database
            result = self.expressions.insert_one(expression)
            expression_id = str(result.inserted_id)
            
            # Update interaction to reference this expression
            self.interactions.update_one(
                {"_id": interaction_obj_id},
                {"$push": {"math_expressions": ObjectId(expression_id)}}
            )
            
            logger.info(f"Stored expression {expression_id} for interaction {interaction_id}")
            return expression_id
            
        except PyMongoError as e:
            logger.error(f"Failed to store expression for interaction {interaction_id}: {e}")
            return None
    
    def get_expression(self, expression_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a mathematical expression by ID.
        
        Args:
            expression_id: ID of the expression
            
        Returns:
            Expression document or None if not found
        """
        try:
            # Convert string ID to ObjectId
            obj_id = ObjectId(expression_id)
            
            # Retrieve from database
            expression = self.expressions.find_one({"_id": obj_id})
            
            if expression:
                # Convert ObjectIds to strings for serialization
                expression["_id"] = str(expression["_id"])
                expression["interaction_id"] = str(expression["interaction_id"])
                return expression
            else:
                logger.warning(f"Expression {expression_id} not found")
                return None
                
        except PyMongoError as e:
            logger.error(f"Failed to get expression {expression_id}: {e}")
            return None
    
    def find_similar_expressions(self, latex_representation: str, 
                                limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar mathematical expressions using text search.
        
        Args:
            latex_representation: LaTeX representation to search for
            limit: Maximum number of expressions to retrieve
            
        Returns:
            List of similar expression documents
        """
        try:
            # Use text search to find similar expressions
            cursor = self.expressions.find(
                {"$text": {"$search": latex_representation}}
            ).limit(limit)
            
            # Convert ObjectIds to strings for serialization
            expressions = []
            for expression in cursor:
                expression["_id"] = str(expression["_id"])
                expression["interaction_id"] = str(expression["interaction_id"])
                expressions.append(expression)
            
            return expressions
            
        except PyMongoError as e:
            logger.error(f"Failed to find similar expressions: {e}")
            return []
    
    def get_expressions_by_domain(self, math_domain: str, 
                                 limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get mathematical expressions by domain.
        
        Args:
            math_domain: Mathematical domain to filter by
            limit: Maximum number of expressions to retrieve
            
        Returns:
            List of expression documents
        """
        try:
            # Retrieve from database
            cursor = self.expressions.find(
                {"math_domain": math_domain}
            ).sort("created_at", -1).limit(limit)
            
            # Convert ObjectIds to strings for serialization
            expressions = []
            for expression in cursor:
                expression["_id"] = str(expression["_id"])
                expression["interaction_id"] = str(expression["interaction_id"])
                expressions.append(expression)
            
            return expressions
            
        except PyMongoError as e:
            logger.error(f"Failed to get expressions for domain {math_domain}: {e}")
            return []

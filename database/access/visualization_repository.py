from typing import Dict, Any, List, Optional, Union
from bson import ObjectId
import os
from datetime import datetime

from database.access.mongodb_wrapper import MongoDBWrapper

class VisualizationRepository:
    """
    Repository for storing and retrieving visualization data.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the Visualization Repository.
        
        Args:
            connection_string: MongoDB connection string (optional)
        """
        self.db = MongoDBWrapper(connection_string).db
        self.visualizations = self.db.visualizations
    
    def store_visualization(
        self, 
        visualization_type: str,
        parameters: Dict[str, Any],
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        interaction_id: Optional[str] = None
    ) -> str:
        """
        Store visualization data in the database.
        
        Args:
            visualization_type: Type of visualization (e.g., 'function_2d')
            parameters: Parameters used to generate the visualization
            file_path: Path to the visualization file
            metadata: Additional metadata about the visualization
            interaction_id: ID of the interaction that generated this visualization
            
        Returns:
            String ID of the stored visualization
        """
        # Create visualization document
        visualization = {
            "visualization_type": visualization_type,
            "parameters": parameters,
            "file_path": file_path,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "file_exists": os.path.exists(file_path)
        }
        
        # Add interaction ID if provided
        if interaction_id:
            visualization["interaction_id"] = interaction_id
        
        # Insert into database
        result = self.visualizations.insert_one(visualization)
        
        return str(result.inserted_id)
    
    def get_visualization(self, visualization_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a visualization by ID.
        
        Args:
            visualization_id: ID of the visualization to retrieve
            
        Returns:
            Visualization document or None if not found
        """
        try:
            result = self.visualizations.find_one({"_id": ObjectId(visualization_id)})
            
            if result:
                # Convert ObjectId to string for serialization
                result["_id"] = str(result["_id"])
                
                # Check if file still exists
                if "file_path" in result:
                    result["file_exists"] = os.path.exists(result["file_path"])
                
                return result
            
            return None
            
        except Exception as e:
            print(f"Error retrieving visualization: {e}")
            return None
    
    def get_visualizations_by_interaction(self, interaction_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all visualizations for a specific interaction.
        
        Args:
            interaction_id: ID of the interaction
            
        Returns:
            List of visualization documents
        """
        try:
            results = list(self.visualizations.find({"interaction_id": interaction_id}))
            
            # Convert ObjectId to string for serialization
            for result in results:
                result["_id"] = str(result["_id"])
                
                # Check if file still exists
                if "file_path" in result:
                    result["file_exists"] = os.path.exists(result["file_path"])
            
            return results
            
        except Exception as e:
            print(f"Error retrieving visualizations: {e}")
            return []
    
    def get_recent_visualizations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent visualizations.
        
        Args:
            limit: Maximum number of visualizations to return
            
        Returns:
            List of visualization documents
        """
        try:
            results = list(self.visualizations.find().sort("created_at", -1).limit(limit))
            
            # Convert ObjectId to string for serialization
            for result in results:
                result["_id"] = str(result["_id"])
                
                # Check if file still exists
                if "file_path" in result:
                    result["file_exists"] = os.path.exists(result["file_path"])
            
            return results
            
        except Exception as e:
            print(f"Error retrieving recent visualizations: {e}")
            return []
    
    def delete_visualization(self, visualization_id: str, delete_file: bool = False) -> bool:
        """
        Delete a visualization from the database.
        
        Args:
            visualization_id: ID of the visualization to delete
            delete_file: Whether to also delete the file
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Get visualization first if we need to delete the file
            file_path = None
            if delete_file:
                visualization = self.visualizations.find_one({"_id": ObjectId(visualization_id)})
                if visualization and "file_path" in visualization:
                    file_path = visualization["file_path"]
            
            # Delete from database
            result = self.visualizations.delete_one({"_id": ObjectId(visualization_id)})
            
            # Delete file if requested
            if delete_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting visualization file: {e}")
            
            return result.deleted_count > 0
            
        except Exception as e:
            print(f"Error deleting visualization: {e}")
            return False

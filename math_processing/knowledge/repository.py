"""
Knowledge Repository

This module provides the database interface for storing and retrieving mathematical
knowledge, including concepts, theorems, and formulas.
"""

import logging
from typing import Dict, List, Any, Optional
import json
import os

logger = logging.getLogger(__name__)

class KnowledgeRepository:
    """
    Repository for storing and retrieving mathematical knowledge.
    
    This class provides methods to interact with the database,
    supporting both in-memory and persistent storage.
    """
    
    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        use_local_storage: bool = True,
        local_storage_path: Optional[str] = None
    ):
        """
        Initialize the knowledge repository.
        
        Args:
            mongodb_uri: MongoDB connection URI
            use_local_storage: Whether to use local file storage
            local_storage_path: Path for local storage files
        """
        self.mongodb_uri = mongodb_uri
        self.use_local_storage = use_local_storage
        self.local_storage_path = local_storage_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data"
        )
        
        # MongoDB collections (initialized if URI is provided)
        self.db = None
        self.concepts_collection = None
        self.theorems_collection = None
        self.formulas_collection = None
        
        # Initialize MongoDB connection if URI is provided
        if mongodb_uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(mongodb_uri)
                self.db = client.get_database("math_knowledge")
                self.concepts_collection = self.db.concepts
                self.theorems_collection = self.db.theorems
                self.formulas_collection = self.db.formulas
                logger.info("Connected to MongoDB")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                if not use_local_storage:
                    raise RuntimeError(f"MongoDB connection failed and local storage is disabled: {e}")
        
        # Ensure local storage directory exists if using local storage
        if use_local_storage:
            os.makedirs(self.local_storage_path, exist_ok=True)
            
            # Create data files if they don't exist
            for filename in ["concepts.json", "theorems.json", "formulas.json"]:
                file_path = os.path.join(self.local_storage_path, filename)
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        json.dump([], f)
            
            logger.info(f"Local storage initialized at {self.local_storage_path}")
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by its ID.
        
        Args:
            concept_id: Identifier for the concept
            
        Returns:
            Concept data or None if not found
        """
        # Try MongoDB first if available
        if self.concepts_collection:
            concept = self.concepts_collection.find_one({"concept_id": concept_id})
            if concept:
                # Convert ObjectId to string for JSON compatibility
                concept["_id"] = str(concept["_id"])
                return concept
        
        # Fall back to local storage if MongoDB failed or is not available
        if self.use_local_storage:
            try:
                file_path = os.path.join(self.local_storage_path, "concepts.json")
                with open(file_path, "r") as f:
                    concepts = json.load(f)
                
                for concept in concepts:
                    if concept.get("concept_id") == concept_id:
                        return concept
            except Exception as e:
                logger.error(f"Error reading from local storage: {e}")
        
        return None
    
    def get_theorem(self, theorem_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a theorem by its ID.
        
        Args:
            theorem_id: Identifier for the theorem
            
        Returns:
            Theorem data or None if not found
        """
        # Try MongoDB first if available
        if self.theorems_collection:
            theorem = self.theorems_collection.find_one({"theorem_id": theorem_id})
            if theorem:
                # Convert ObjectId to string for JSON compatibility
                theorem["_id"] = str(theorem["_id"])
                return theorem
        
        # Fall back to local storage if MongoDB failed or is not available
        if self.use_local_storage:
            try:
                file_path = os.path.join(self.local_storage_path, "theorems.json")
                with open(file_path, "r") as f:
                    theorems = json.load(f)
                
                for theorem in theorems:
                    if theorem.get("theorem_id") == theorem_id:
                        return theorem
            except Exception as e:
                logger.error(f"Error reading from local storage: {e}")
        
        return None
    
    def get_formula(self, formula_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a formula by its ID.
        
        Args:
            formula_id: Identifier for the formula
            
        Returns:
            Formula data or None if not found
        """
        # Try MongoDB first if available
        if self.formulas_collection:
            formula = self.formulas_collection.find_one({"formula_id": formula_id})
            if formula:
                # Convert ObjectId to string for JSON compatibility
                formula["_id"] = str(formula["_id"])
                return formula
        
        # Fall back to local storage if MongoDB failed or is not available
        if self.use_local_storage:
            try:
                file_path = os.path.join(self.local_storage_path, "formulas.json")
                with open(file_path, "r") as f:
                    formulas = json.load(f)
                
                for formula in formulas:
                    if formula.get("formula_id") == formula_id:
                        return formula
            except Exception as e:
                logger.error(f"Error reading from local storage: {e}")
        
        return None
    
    def add_concept(self, concept: Dict[str, Any]) -> bool:
        """
        Add a new concept.
        
        Args:
            concept: Concept data
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure concept_id is present
        if "concept_id" not in concept:
            logger.error("Cannot add concept: missing concept_id")
            return False
        
        # Try MongoDB first if available
        success = False
        if self.concepts_collection:
            try:
                # Check if concept already exists
                existing = self.concepts_collection.find_one({"concept_id": concept["concept_id"]})
                if existing:
                    # Update existing concept
                    self.concepts_collection.update_one(
                        {"concept_id": concept["concept_id"]},
                        {"$set": concept}
                    )
                else:
                    # Insert new concept
                    self.concepts_collection.insert_one(concept)
                success = True
            except Exception as e:
                logger.error(f"Error adding concept to MongoDB: {e}")
        
        # Add to local storage if MongoDB failed or is not available
        if self.use_local_storage:
            try:
                file_path = os.path.join(self.local_storage_path, "concepts.json")
                
                # Read existing concepts
                with open(file_path, "r") as f:
                    concepts = json.load(f)
                
                # Check if concept already exists
                updated = False
                for i, existing in enumerate(concepts):
                    if existing.get("concept_id") == concept["concept_id"]:
                        concepts[i] = concept
                        updated = True
                        break
                
                # Add if not updated
                if not updated:
                    concepts.append(concept)
                
                # Write back to file
                with open(file_path, "w") as f:
                    json.dump(concepts, f, indent=2)
                
                success = True
            except Exception as e:
                logger.error(f"Error adding concept to local storage: {e}")
        
        return success
    
    def add_theorem(self, theorem: Dict[str, Any]) -> bool:
        """
        Add a new theorem.
        
        Args:
            theorem: Theorem data
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure theorem_id is present
        if "theorem_id" not in theorem:
            logger.error("Cannot add theorem: missing theorem_id")
            return False
        
        # Try MongoDB first if available
        success = False
        if self.theorems_collection:
            try:
                # Check if theorem already exists
                existing = self.theorems_collection.find_one({"theorem_id": theorem["theorem_id"]})
                if existing:
                    # Update existing theorem
                    self.theorems_collection.update_one(
                        {"theorem_id": theorem["theorem_id"]},
                        {"$set": theorem}
                    )
                else:
                    # Insert new theorem
                    self.theorems_collection.insert_one(theorem)
                success = True
            except Exception as e:
                logger.error(f"Error adding theorem to MongoDB: {e}")
        
        # Add to local storage if MongoDB failed or is not available
        if self.use_local_storage:
            try:
                file_path = os.path.join(self.local_storage_path, "theorems.json")
                
                # Read existing theorems
                with open(file_path, "r") as f:
                    theorems = json.load(f)
                
                # Check if theorem already exists
                updated = False
                for i, existing in enumerate(theorems):
                    if existing.get("theorem_id") == theorem["theorem_id"]:
                        theorems[i] = theorem
                        updated = True
                        break
                
                # Add if not updated
                if not updated:
                    theorems.append(theorem)
                
                # Write back to file
                with open(file_path, "w") as f:
                    json.dump(theorems, f, indent=2)
                
                success = True
            except Exception as e:
                logger.error(f"Error adding theorem to local storage: {e}")
        
        return success
    
    def add_formula(self, formula: Dict[str, Any]) -> bool:
        """
        Add a new formula.
        
        Args:
            formula: Formula data
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure formula_id is present
        if "formula_id" not in formula:
            logger.error("Cannot add formula: missing formula_id")
            return False
        
        # Try MongoDB first if available
        success = False
        if self.formulas_collection:
            try:
                # Check if formula already exists
                existing = self.formulas_collection.find_one({"formula_id": formula["formula_id"]})
                if existing:
                    # Update existing formula
                    self.formulas_collection.update_one(
                        {"formula_id": formula["formula_id"]},
                        {"$set": formula}
                    )
                else:
                    # Insert new formula
                    self.formulas_collection.insert_one(formula)
                success = True
            except Exception as e:
                logger.error(f"Error adding formula to MongoDB: {e}")
        
        # Add to local storage if MongoDB failed or is not available
        if self.use_local_storage:
            try:
                file_path = os.path.join(self.local_storage_path, "formulas.json")
                
                # Read existing formulas
                with open(file_path, "r") as f:
                    formulas = json.load(f)
                
                # Check if formula already exists
                updated = False
                for i, existing in enumerate(formulas):
                    if existing.get("formula_id") == formula["formula_id"]:
                        formulas[i] = formula
                        updated = True
                        break
                
                # Add if not updated
                if not updated:
                    formulas.append(formula)
                
                # Write back to file
                with open(file_path, "w") as f:
                    json.dump(formulas, f, indent=2)
                
                success = True
            except Exception as e:
                logger.error(f"Error adding formula to local storage: {e}")
        
        return success
    
    def search_concepts(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for concepts based on criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching concepts
        """
        results = []
        
        # Try MongoDB first if available
        if self.concepts_collection:
            try:
                # Convert search criteria to MongoDB query
                query = self._convert_criteria_to_query(criteria)
                
                # Execute the query
                cursor = self.concepts_collection.find(query)
                
                # Process results
                for concept in cursor:
                    # Convert ObjectId to string for JSON compatibility
                    concept["_id"] = str(concept["_id"])
                    results.append(concept)
            except Exception as e:
                logger.error(f"Error searching concepts in MongoDB: {e}")
        
        # Search in local storage if MongoDB failed or is not available
        if self.use_local_storage and not results:
            try:
                file_path = os.path.join(self.local_storage_path, "concepts.json")
                
                # Read all concepts
                with open(file_path, "r") as f:
                    concepts = json.load(f)
                
                # Filter based on criteria
                results = self._filter_by_criteria(concepts, criteria)
            except Exception as e:
                logger.error(f"Error searching concepts in local storage: {e}")
        
        return results
    
    def search_theorems(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for theorems based on criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching theorems
        """
        results = []
        
        # Try MongoDB first if available
        if self.theorems_collection:
            try:
                # Convert search criteria to MongoDB query
                query = self._convert_criteria_to_query(criteria)
                
                # Execute the query
                cursor = self.theorems_collection.find(query)
                
                # Process results
                for theorem in cursor:
                    # Convert ObjectId to string for JSON compatibility
                    theorem["_id"] = str(theorem["_id"])
                    results.append(theorem)
            except Exception as e:
                logger.error(f"Error searching theorems in MongoDB: {e}")
        
        # Search in local storage if MongoDB failed or is not available
        if self.use_local_storage and not results:
            try:
                file_path = os.path.join(self.local_storage_path, "theorems.json")
                
                # Read all theorems
                with open(file_path, "r") as f:
                    theorems = json.load(f)
                
                # Filter based on criteria
                results = self._filter_by_criteria(theorems, criteria)
            except Exception as e:
                logger.error(f"Error searching theorems in local storage: {e}")
        
        return results
    
    def search_formulas(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for formulas based on criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching formulas
        """
        results = []
        
        # Try MongoDB first if available
        if self.formulas_collection:
            try:
                # Convert search criteria to MongoDB query
                query = self._convert_criteria_to_query(criteria)
                
                # Execute the query
                cursor = self.formulas_collection.find(query)
                
                # Process results
                for formula in cursor:
                    # Convert ObjectId to string for JSON compatibility
                    formula["_id"] = str(formula["_id"])
                    results.append(formula)
            except Exception as e:
                logger.error(f"Error searching formulas in MongoDB: {e}")
        
        # Search in local storage if MongoDB failed or is not available
        if self.use_local_storage and not results:
            try:
                file_path = os.path.join(self.local_storage_path, "formulas.json")
                
                # Read all formulas
                with open(file_path, "r") as f:
                    formulas = json.load(f)
                
                # Filter based on criteria
                results = self._filter_by_criteria(formulas, criteria)
            except Exception as e:
                logger.error(f"Error searching formulas in local storage: {e}")
        
        return results
    
    def _convert_criteria_to_query(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert search criteria to a MongoDB query.
        
        Args:
            criteria: Search criteria
            
        Returns:
            MongoDB query
        """
        query = {}
        
        # Process each criterion
        for key, value in criteria.items():
            if key == "domain":
                query["domain"] = value
            elif key == "name_contains":
                query["name"] = {"$regex": value, "$options": "i"}
            elif key == "related_concepts":
                query["related_concepts"] = {"$in": value if isinstance(value, list) else [value]}
            else:
                # Default to exact match
                query[key] = value
        
        return query
    
    def _filter_by_criteria(self, items: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter a list of items based on search criteria.
        
        Args:
            items: List of items to filter
            criteria: Search criteria
            
        Returns:
            Filtered list of items
        """
        results = []
        
        for item in items:
            match = True
            
            # Check each criterion
            for key, value in criteria.items():
                if key == "domain":
                    if item.get("domain") != value:
                        match = False
                        break
                elif key == "name_contains":
                    if value.lower() not in item.get("name", "").lower():
                        match = False
                        break
                elif key == "related_concepts":
                    if not any(concept in item.get("related_concepts", []) for concept in 
                              (value if isinstance(value, list) else [value])):
                        match = False
                        break
                else:
                    # Default to exact match
                    if item.get(key) != value:
                        match = False
                        break
            
            if match:
                results.append(item)
        
        return results

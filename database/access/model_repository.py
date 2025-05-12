"""
Model repository for storing model metadata and configurations.

This module provides functionality to store and retrieve model metadata,
configurations, and performance metrics in the database.
"""
import logging
import datetime
from typing import Dict, Any, List, Optional, Union
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId

logger = logging.getLogger(__name__)

class ModelRepository:
    """Repository for model metadata and configurations."""
    
    def __init__(self, connection_string: str, db_name: str = "math_llm_system"):
        """
        Initialize the model repository.
        
        Args:
            connection_string: MongoDB connection string
            db_name: Database name
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.models = self.db.models
        self.model_versions = self.db.model_versions
        self.model_configs = self.db.model_configs
        self.model_metrics = self.db.model_metrics
        
        # Ensure indexes
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Create necessary indexes."""
        # Models collection
        self.models.create_index([("name", ASCENDING)], unique=True)
        
        # Model versions collection
        self.model_versions.create_index([("model_id", ASCENDING), ("version", DESCENDING)])
        self.model_versions.create_index([("model_id", ASCENDING), ("status", ASCENDING)])
        
        # Model configs collection
        self.model_configs.create_index([("model_id", ASCENDING), ("version_id", ASCENDING)])
        
        # Model metrics collection
        self.model_metrics.create_index([("model_id", ASCENDING), ("version_id", ASCENDING)])
        self.model_metrics.create_index([("domain", ASCENDING)])
        self.model_metrics.create_index([("created_at", DESCENDING)])
    
    def create_model(self, name: str, description: str, model_type: str) -> str:
        """
        Create a new model entry.
        
        Args:
            name: Model name
            description: Model description
            model_type: Type of model (e.g., "llm", "ocr", "computation")
            
        Returns:
            ID of the created model
        """
        model = {
            "name": name,
            "description": description,
            "model_type": model_type,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        try:
            result = self.models.insert_one(model)
            model_id = str(result.inserted_id)
            logger.info(f"Created model: {name} with ID: {model_id}")
            return model_id
        except Exception as e:
            logger.error(f"Error creating model {name}: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model document or None if not found
        """
        try:
            model = self.models.find_one({"_id": ObjectId(model_id)})
            if model:
                model["_id"] = str(model["_id"])
            return model
        except Exception as e:
            logger.error(f"Error retrieving model {model_id}: {e}")
            return None
    
    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a model by name.
        
        Args:
            name: Model name
            
        Returns:
            Model document or None if not found
        """
        try:
            model = self.models.find_one({"name": name})
            if model:
                model["_id"] = str(model["_id"])
            return model
        except Exception as e:
            logger.error(f"Error retrieving model by name {name}: {e}")
            return None
    
    def create_model_version(
        self,
        model_id: str,
        version: str,
        source_url: str,
        local_path: Optional[str] = None,
        status: str = "pending",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new model version.
        
        Args:
            model_id: Model ID
            version: Version string
            source_url: URL where the model was downloaded from
            local_path: Local path where the model is stored
            status: Status of the model version
            metadata: Additional metadata
            
        Returns:
            ID of the created model version
        """
        model_version = {
            "model_id": model_id,
            "version": version,
            "source_url": source_url,
            "local_path": local_path,
            "status": status,
            "metadata": metadata or {},
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        try:
            result = self.model_versions.insert_one(model_version)
            version_id = str(result.inserted_id)
            logger.info(f"Created model version: {version} for model ID: {model_id}")
            return version_id
        except Exception as e:
            logger.error(f"Error creating model version {version} for model {model_id}: {e}")
            raise
    
    def update_model_version_status(
        self,
        version_id: str,
        status: str,
        local_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update the status of a model version.
        
        Args:
            version_id: Model version ID
            status: New status
            local_path: Updated local path
            metadata: Additional metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        update = {
            "$set": {
                "status": status,
                "updated_at": datetime.datetime.utcnow()
            }
        }
        
        if local_path is not None:
            update["$set"]["local_path"] = local_path
        
        if metadata is not None:
            for key, value in metadata.items():
                update["$set"][f"metadata.{key}"] = value
        
        try:
            result = self.model_versions.update_one(
                {"_id": ObjectId(version_id)},
                update
            )
            logger.info(f"Updated model version {version_id} status to {status}")
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating model version status {version_id}: {e}")
            return False
    
    def get_model_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model version by ID.
        
        Args:
            version_id: Model version ID
            
        Returns:
            Model version document or None if not found
        """
        try:
            version = self.model_versions.find_one({"_id": ObjectId(version_id)})
            if version:
                version["_id"] = str(version["_id"])
                version["model_id"] = str(version["model_id"])
            return version
        except Exception as e:
            logger.error(f"Error retrieving model version {version_id}: {e}")
            return None
    
    def get_latest_model_version(self, model_id: str, status: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model.
        
        Args:
            model_id: Model ID
            status: Optional status filter
            
        Returns:
            Latest model version document or None if not found
        """
        try:
            query = {"model_id": model_id}
            if status:
                query["status"] = status
            
            version = self.model_versions.find_one(
                query,
                sort=[("created_at", DESCENDING)]
            )
            
            if version:
                version["_id"] = str(version["_id"])
                version["model_id"] = str(version["model_id"])
            
            return version
        except Exception as e:
            logger.error(f"Error retrieving latest model version for model {model_id}: {e}")
            return None
    
    def create_model_config(
        self,
        model_id: str,
        version_id: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a model configuration.
        
        Args:
            model_id: Model ID
            version_id: Model version ID
            config: Configuration dictionary
            name: Configuration name
            description: Configuration description
            
        Returns:
            ID of the created configuration
        """
        model_config = {
            "model_id": model_id,
            "version_id": version_id,
            "name": name or f"Config {datetime.datetime.utcnow().isoformat()}",
            "description": description,
            "config": config,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        try:
            result = self.model_configs.insert_one(model_config)
            config_id = str(result.inserted_id)
            logger.info(f"Created model config for model {model_id}, version {version_id}")
            return config_id
        except Exception as e:
            logger.error(f"Error creating model config: {e}")
            raise
    
    def get_model_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            Configuration document or None if not found
        """
        try:
            config = self.model_configs.find_one({"_id": ObjectId(config_id)})
            if config:
                config["_id"] = str(config["_id"])
                config["model_id"] = str(config["model_id"])
                config["version_id"] = str(config["version_id"])
            return config
        except Exception as e:
            logger.error(f"Error retrieving model config {config_id}: {e}")
            return None
    
    def get_latest_model_config(self, model_id: str, version_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest configuration for a model or model version.
        
        Args:
            model_id: Model ID
            version_id: Optional model version ID
            
        Returns:
            Latest configuration document or None if not found
        """
        try:
            query = {"model_id": model_id}
            if version_id:
                query["version_id"] = version_id
            
            config = self.model_configs.find_one(
                query,
                sort=[("created_at", DESCENDING)]
            )
            
            if config:
                config["_id"] = str(config["_id"])
                config["model_id"] = str(config["model_id"])
                config["version_id"] = str(config["version_id"])
            
            return config
        except Exception as e:
            logger.error(f"Error retrieving latest model config for model {model_id}: {e}")
            return None
    
    def record_model_metrics(
        self,
        model_id: str,
        version_id: str,
        metrics: Dict[str, Any],
        domain: Optional[str] = None,
        task: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> str:
        """
        Record metrics for a model version.
        
        Args:
            model_id: Model ID
            version_id: Model version ID
            metrics: Metrics dictionary
            domain: Optional domain (e.g., "algebra", "calculus")
            task: Optional task type
            dataset: Optional dataset name
            
        Returns:
            ID of the created metrics record
        """
        metrics_record = {
            "model_id": model_id,
            "version_id": version_id,
            "domain": domain,
            "task": task,
            "dataset": dataset,
            "metrics": metrics,
            "created_at": datetime.datetime.utcnow()
        }
        
        try:
            result = self.model_metrics.insert_one(metrics_record)
            metrics_id = str(result.inserted_id)
            logger.info(f"Recorded metrics for model {model_id}, version {version_id}")
            return metrics_id
        except Exception as e:
            logger.error(f"Error recording model metrics: {e}")
            raise
    
    def get_model_metrics(
        self,
        model_id: str,
        version_id: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get metrics for a model.
        
        Args:
            model_id: Model ID
            version_id: Optional model version ID
            domain: Optional domain filter
            limit: Maximum number of records to return
            
        Returns:
            List of metrics records
        """
        try:
            query = {"model_id": model_id}
            if version_id:
                query["version_id"] = version_id
            if domain:
                query["domain"] = domain
            
            cursor = self.model_metrics.find(
                query,
                sort=[("created_at", DESCENDING)]
            ).limit(limit)
            
            metrics = []
            for record in cursor:
                record["_id"] = str(record["_id"])
                record["model_id"] = str(record["model_id"])
                record["version_id"] = str(record["version_id"])
                metrics.append(record)
            
            return metrics
        except Exception as e:
            logger.error(f"Error retrieving model metrics for model {model_id}: {e}")
            return []

"""
Database schema manager for the Mathematical Multimodal LLM System.

This module handles database schema initialization, migrations, and validation.
"""
import logging
import datetime
from typing import Dict, Any, List, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT, IndexModel

from config.mongodb_config import get_mongo_config, get_collection_name

logger = logging.getLogger(__name__)

class SchemaManager:
    """
    Handles database schema management including initialization, migrations, and validation.
    """
    
    def __init__(self, connection_string: Optional[str] = None, db_name: Optional[str] = None):
        """
        Initialize the schema manager.
        
        Args:
            connection_string: Optional MongoDB connection string override
            db_name: Optional database name override
        """
        # Get MongoDB configuration
        mongo_config = get_mongo_config()
        
        # Use overrides if provided
        if connection_string:
            mongo_config["uri"] = connection_string
        
        if db_name:
            mongo_config["database"] = db_name
        
        # Connect to database
        self.client = MongoClient(mongo_config["uri"], **mongo_config.get("options", {}))
        self.db = self.client[mongo_config["database"]]
        
        # Define schema versions
        self.current_schema_version = "1.0.0"
        
        # Initialize schema version collection
        self.schema_versions = self.db["schema_versions"]
    
    def initialize_schema(self, force: bool = False) -> bool:
        """
        Initialize the database schema.
        
        Args:
            force: Force initialization even if schema is already initialized
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Check if schema is already initialized
            if not force and self._is_schema_initialized():
                logger.info("Schema already initialized, skipping initialization")
                return True
            
            logger.info("Initializing database schema...")
            
            # Create collections with validators
            self._create_collections()
            
            # Create indexes
            self._create_indexes()
            
            # Record schema version
            self._record_schema_version(self.current_schema_version)
            
            logger.info(f"Schema initialization complete (version {self.current_schema_version})")
            return True
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            return False
    
    def _is_schema_initialized(self) -> bool:
        """
        Check if the schema is already initialized.
        
        Returns:
            True if schema is initialized, False otherwise
        """
        # Check if schema version collection exists and has records
        return self.schema_versions.count_documents({}) > 0
    
    def _record_schema_version(self, version: str) -> None:
        """
        Record a schema version.
        
        Args:
            version: Schema version
        """
        self.schema_versions.insert_one({
            "version": version,
            "applied_at": datetime.datetime.utcnow(),
            "status": "completed"
        })
    
    def _create_collections(self) -> None:
        """Create collections with validators."""
        # Define collection validators
        validators = {
            "users": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["username", "email", "created_at"],
                    "properties": {
                        "username": {"bsonType": "string"},
                        "email": {"bsonType": "string"},
                        "created_at": {"bsonType": "date"}
                    }
                }
            },
            "conversations": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "created_at"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "title": {"bsonType": "string"},
                        "created_at": {"bsonType": "date"},
                        "updated_at": {"bsonType": "date"}
                    }
                }
            },
            "math_expressions": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["latex_representation", "created_at"],
                    "properties": {
                        "latex_representation": {"bsonType": "string"},
                        "symbolic_representation": {"bsonType": "string"},
                        "domain": {"bsonType": "string"},
                        "conversation_id": {"bsonType": "string"},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        }
        
        # Create collections
        for collection_key, validator in validators.items():
            collection_name = get_collection_name(collection_key)
            
            try:
                # Create collection with validator
                self.db.create_collection(
                    collection_name,
                    validator=validator,
                    validationLevel="moderate"
                )
                logger.info(f"Created collection: {collection_name} with validator")
            except Exception as e:
                # Collection may already exist
                logger.info(f"Collection {collection_name} already exists or could not be created: {e}")
                
                # Try to apply validator to existing collection
                try:
                    self.db.command({
                        "collMod": collection_name,
                        "validator": validator,
                        "validationLevel": "moderate"
                    })
                    logger.info(f"Applied validator to existing collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to apply validator to collection {collection_name}: {e}")
    
    def _create_indexes(self) -> None:
        """Create indexes for collections."""
        # Define indexes
        indexes = {
            "users": [
                IndexModel([("username", ASCENDING)], unique=True),
                IndexModel([("email", ASCENDING)], unique=True),
                IndexModel([("created_at", DESCENDING)])
            ],
            "conversations": [
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("updated_at", DESCENDING)]),
                IndexModel([("title", TEXT)])
            ],
            "interactions": [
                IndexModel([("conversation_id", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)])
            ],
            "math_expressions": [
                IndexModel([("conversation_id", ASCENDING)]),
                IndexModel([("domain", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("latex_representation", TEXT)])
            ],
            "handwritten_inputs": [
                IndexModel([("conversation_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)])
            ],
            "visualizations": [
                IndexModel([("conversation_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)])
            ],
            "math_knowledge": [
                IndexModel([("domain", ASCENDING)]),
                IndexModel([("concept_name", TEXT)]),
                IndexModel([("related_concepts", ASCENDING)])
            ],
            "models": [
                IndexModel([("name", ASCENDING)], unique=True),
                IndexModel([("model_type", ASCENDING)])
            ],
            "model_versions": [
                IndexModel([("model_id", ASCENDING), ("version", ASCENDING)], unique=True),
                IndexModel([("status", ASCENDING)])
            ],
            "workflows": [
                IndexModel([("state", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("updated_at", DESCENDING)])
            ],
            "agent_registry": [
                IndexModel([("capabilities", ASCENDING)]),
                IndexModel([("status", ASCENDING)])
            ]
        }
        
        # Create indexes
        for collection_key, collection_indexes in indexes.items():
            collection_name = get_collection_name(collection_key)
            collection = self.db[collection_name]
            
            try:
                # Create indexes
                collection.create_indexes(collection_indexes)
                logger.info(f"Created indexes for collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to create indexes for collection {collection_name}: {e}")
    
    def get_current_schema_version(self) -> str:
        """
        Get the current schema version.
        
        Returns:
            Current schema version
        """
        # Get most recent schema version
        latest_version = self.schema_versions.find_one(
            {"status": "completed"},
            sort=[("applied_at", DESCENDING)]
        )
        
        if latest_version:
            return latest_version["version"]
        else:
            return "0.0.0"  # No version found
    
    def check_data_integrity(self) -> Dict[str, Any]:
        """
        Check data integrity of the database.
        
        Returns:
            Integrity check results
        """
        results = {
            "checked_at": datetime.datetime.utcnow(),
            "schema_version": self.get_current_schema_version(),
            "collections": {},
            "issues": []
        }
        
        # List of collections to check
        collections_to_check = [
            "users", "conversations", "interactions", "math_expressions",
            "handwritten_inputs", "visualizations", "math_knowledge",
            "models", "model_versions", "workflows", "agent_registry"
        ]
        
        # Check each collection
        for collection_key in collections_to_check:
            collection_name = get_collection_name(collection_key)
            collection = self.db[collection_name]
            
            # Collection statistics
            stats = {
                "count": collection.count_documents({}),
                "indexes": len(collection.index_information()),
                "size": 0  # Will be updated if available
            }
            
            # Get collection size
            try:
                collection_stats = self.db.command("collStats", collection_name)
                stats["size"] = collection_stats.get("size", 0)
                stats["storage_size"] = collection_stats.get("storageSize", 0)
            except Exception as e:
                results["issues"].append({
                    "collection": collection_name,
                    "type": "stats_error",
                    "message": str(e)
                })
            
            # Check for missing indexes
            required_indexes = []
            for idx in indexes.get(collection_key, []):
                key_pattern = idx.document["key"]
                index_name = "_".join([f"{k}_{v}" for k, v in key_pattern])
                required_indexes.append(index_name)
            
            actual_indexes = list(collection.index_information().keys())
            missing_indexes = [idx for idx in required_indexes if idx not in actual_indexes]
            
            if missing_indexes:
                results["issues"].append({
                    "collection": collection_name,
                    "type": "missing_indexes",
                    "indexes": missing_indexes
                })
            
            # Store collection results
            results["collections"][collection_name] = stats
        
        # Add overall database stats
        try:
            db_stats = self.db.command("dbStats")
            results["database"] = {
                "collections": db_stats.get("collections", 0),
                "objects": db_stats.get("objects", 0),
                "data_size": db_stats.get("dataSize", 0),
                "storage_size": db_stats.get("storageSize", 0),
                "indexes": db_stats.get("indexes", 0),
                "index_size": db_stats.get("indexSize", 0)
            }
        except Exception as e:
            results["issues"].append({
                "type": "db_stats_error",
                "message": str(e)
            })
        
        return results
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """
        Perform database maintenance tasks.
        
        Returns:
            Maintenance results
        """
        results = {
            "performed_at": datetime.datetime.utcnow(),
            "actions": [],
            "issues": []
        }
        
        try:
            # Compact collections
            collections = self.db.list_collection_names()
            for collection_name in collections:
                try:
                    self.db.command("compact", collection_name)
                    results["actions"].append({
                        "collection": collection_name,
                        "action": "compact",
                        "success": True
                    })
                except Exception as e:
                    results["issues"].append({
                        "collection": collection_name,
                        "action": "compact",
                        "success": False,
                        "message": str(e)
                    })
            
            # Clean up old workflow documents
            workflow_collection = self.db[get_collection_name("workflows")]
            thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
            
            delete_result = workflow_collection.delete_many({
                "state": {"$in": ["completed", "failed"]},
                "updated_at": {"$lt": thirty_days_ago}
            })
            
            results["actions"].append({
                "collection": get_collection_name("workflows"),
                "action": "cleanup_old_workflows",
                "count": delete_result.deleted_count,
                "success": True
            })
            
            # Add more maintenance tasks as needed
            
        except Exception as e:
            results["issues"].append({
                "action": "general_maintenance",
                "success": False,
                "message": str(e)
            })
        
        return results

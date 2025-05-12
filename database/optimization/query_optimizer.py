"""
Database query optimization module.
Provides query analysis, optimization, and performance monitoring for MongoDB operations.
"""
import time
import logging
import functools
import json
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from pymongo import MongoClient, IndexModel
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from bson import ObjectId, json_util
import numpy as np
from math_llm_system.orchestration.monitoring.logger import get_logger

logger = get_logger("database.query_optimizer")

class QueryOptimizer:
    """
    Query optimization engine for MongoDB operations.
    Provides query analysis, index recommendations, and query rewriting.
    """
    
    def __init__(self, client: MongoClient):
        """
        Initialize query optimizer.
        
        Args:
            client: MongoDB client
        """
        self.client = client
        self.query_stats = {}
        self.collection_stats = {}
        self.index_recommendations = {}
        self.explained_queries = {}
    
    def analyze_query(self, collection: Collection, query: Dict[str, Any], 
                      projection: Optional[Dict[str, Any]] = None, 
                      sort: Optional[List[Tuple[str, int]]] = None) -> Dict[str, Any]:
        """
        Analyze a query for performance characteristics.
        
        Args:
            collection: MongoDB collection
            query: Query filter
            projection: Query projection
            sort: Query sort
            
        Returns:
            Dictionary with query analysis
        """
        # Get explain plan
        explain_plan = self._get_explain_plan(collection, query, projection, sort)
        
        # Track query for statistics
        query_key = self._get_query_key(collection.name, query, projection, sort)
        
        if query_key not in self.query_stats:
            self.query_stats[query_key] = {
                "count": 0,
                "total_time": 0,
                "max_time": 0,
                "min_time": float('inf'),
                "avg_time": 0,
                "last_seen": time.time(),
                "collection": collection.name,
                "query": self._stringify_for_storage(query),
                "projection": self._stringify_for_storage(projection) if projection else None,
                "sort": sort
            }
        
        # Run query with timing
        start_time = time.time()
        result = collection.find(query, projection=projection)
        if sort:
            result = result.sort(sort)
        
        # Force execution of cursor to get timing
        result_count = 0
        for _ in result:
            result_count += 1
            
        query_time = time.time() - start_time
        
        # Update statistics
        stats = self.query_stats[query_key]
        stats["count"] += 1
        stats["total_time"] += query_time
        stats["max_time"] = max(stats["max_time"], query_time)
        stats["min_time"] = min(stats["min_time"], query_time)
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["last_seen"] = time.time()
        stats["last_count"] = result_count
        
        # Store explain plan
        self.explained_queries[query_key] = explain_plan
        
        # Check if query is inefficient and generate index recommendations
        needs_index = self._check_needs_index(explain_plan)
        if needs_index:
            recommended_indexes = self._recommend_indexes(collection, query, sort)
            if recommended_indexes:
                self.index_recommendations[query_key] = recommended_indexes
        
        # Return analysis
        return {
            "query_time": query_time,
            "result_count": result_count,
            "needs_index": needs_index,
            "recommended_indexes": self.index_recommendations.get(query_key, []),
            "query_stats": self.query_stats[query_key],
            "explain_plan": explain_plan
        }
    
    def optimize_query(self, collection: Collection, query: Dict[str, Any],
                      projection: Optional[Dict[str, Any]] = None,
                      sort: Optional[List[Tuple[str, int]]] = None) -> Dict[str, Any]:
        """
        Optimize a query for better performance.
        
        Args:
            collection: MongoDB collection
            query: Query filter
            projection: Query projection
            sort: Query sort
            
        Returns:
            Dictionary with optimized query elements
        """
        optimized_query = query.copy()
        optimized_projection = projection.copy() if projection else None
        optimized_sort = sort.copy() if sort else None
        
        # Apply query optimizations
        optimized_query = self._optimize_query_filter(optimized_query)
        
        # Apply projection optimizations
        if optimized_projection:
            optimized_projection = self._optimize_projection(optimized_projection)
        
        # Check for covered query potential
        can_be_covered = self._check_covered_query_potential(
            collection, optimized_query, optimized_projection, optimized_sort
        )
        
        return {
            "optimized_query": optimized_query,
            "optimized_projection": optimized_projection,
            "optimized_sort": optimized_sort,
            "can_be_covered": can_be_covered
        }
    
    def apply_recommended_indexes(self, collection: Collection, 
                                 query_key: Optional[str] = None,
                                 background: bool = True) -> List[str]:
        """
        Apply recommended indexes to a collection.
        
        Args:
            collection: MongoDB collection
            query_key: Specific query key to apply indexes for (or all if None)
            background: Whether to create indexes in the background
            
        Returns:
            List of created index names
        """
        created_indexes = []
        
        # Get recommendations to apply
        if query_key and query_key in self.index_recommendations:
            recommendations = self.index_recommendations[query_key]
        else:
            # Combine all recommendations for the collection
            recommendations = []
            for qk, indexes in self.index_recommendations.items():
                if qk.startswith(f"collection:{collection.name}:"):
                    recommendations.extend(indexes)
        
        if not recommendations:
            return created_indexes
        
        # De-duplicate index recommendations
        unique_indexes = {}
        for index in recommendations:
            index_key = json.dumps(index["keys"], sort_keys=True)
            if index_key not in unique_indexes:
                unique_indexes[index_key] = index
        
        # Create each recommended index
        for index in unique_indexes.values():
            try:
                # Check if index already exists
                existing_indexes = collection.index_information()
                index_exists = False
                
                for existing_name, existing_info in existing_indexes.items():
                    if existing_name == "_id_":  # Skip _id index
                        continue
                        
                    existing_keys = [(key, direction) for key, direction in existing_info["key"]]
                    if existing_keys == index["keys"]:
                        index_exists = True
                        break
                
                if index_exists:
                    logger.info(f"Index already exists: {index['keys']} on {collection.name}")
                    continue
                
                # Create the index
                index_model = IndexModel(
                    index["keys"],
                    background=background,
                    name=index.get("name")
                )
                
                result = collection.create_indexes([index_model])
                created_indexes.append(result[0])
                logger.info(f"Created index {result[0]} on {collection.name}: {index['keys']}")
                
            except Exception as e:
                logger.error(f"Error creating index on {collection.name}: {e}")
        
        return created_indexes
    
    def get_collection_statistics(self, collection: Collection) -> Dict[str, Any]:
        """
        Get detailed statistics for a collection.
        
        Args:
            collection: MongoDB collection
            
        Returns:
            Dictionary of collection statistics
        """
        try:
            # Get collection stats from MongoDB
            stats = collection.database.command("collStats", collection.name)
            
            # Get index sizes and usage statistics
            index_stats = collection.database.command(
                "aggregate", 
                collection.name, 
                pipeline=[
                    {"$indexStats": {}}
                ],
                cursor={}
            )
            
            # Process index usage statistics
            index_usage = []
            if "cursor" in index_stats and "firstBatch" in index_stats["cursor"]:
                for index_stat in index_stats["cursor"]["firstBatch"]:
                    index_usage.append({
                        "name": index_stat.get("name"),
                        "key": index_stat.get("key"),
                        "ops": index_stat.get("accesses", {}).get("ops", 0),
                        "since": index_stat.get("accesses", {}).get("since")
                    })
            
            # Store in collection stats
            self.collection_stats[collection.name] = {
                "count": stats.get("count", 0),
                "size": stats.get("size", 0),
                "avg_doc_size": stats.get("avgObjSize", 0),
                "storage_size": stats.get("storageSize", 0),
                "index_size": stats.get("totalIndexSize", 0),
                "index_count": len(stats.get("indexSizes", {})),
                "index_sizes": stats.get("indexSizes", {}),
                "index_usage": index_usage,
                "wired_tiger": stats.get("wiredTiger", {}),
                "timestamp": time.time()
            }
            
            return self.collection_stats[collection.name]
            
        except Exception as e:
            logger.error(f"Error getting statistics for {collection.name}: {e}")
            return {}
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report with recommendations.
        
        Returns:
            Dictionary with optimization report and recommendations
        """
        report = {
            "timestamp": time.time(),
            "collection_stats": self.collection_stats,
            "slow_queries": [],
            "index_recommendations": [],
            "unused_indexes": [],
            "collection_recommendations": {}
        }
        
        # Find slow queries
        for query_key, stats in self.query_stats.items():
            if stats["avg_time"] > 0.1:  # Queries taking > 100ms on average
                report["slow_queries"].append({
                    "key": query_key,
                    "collection": stats["collection"],
                    "query": stats["query"],
                    "avg_time": stats["avg_time"],
                    "count": stats["count"],
                    "recommended_indexes": self.index_recommendations.get(query_key, [])
                })
        
        # Sort slow queries by average time (descending)
        report["slow_queries"].sort(key=lambda x: x["avg_time"], reverse=True)
        
        # Identify unused indexes
        for collection_name, stats in self.collection_stats.items():
            unused_indexes = []
            for index in stats.get("index_usage", []):
                if index["name"] != "_id_" and index["ops"] == 0:
                    unused_indexes.append(index["name"])
            
            if unused_indexes:
                report["unused_indexes"].append({
                    "collection": collection_name,
                    "indexes": unused_indexes
                })
        
        # Collection-specific recommendations
        for collection_name, stats in self.collection_stats.items():
            recommendations = []
            
            # Check for large average document size
            if stats.get("avg_doc_size", 0) > 64000:  # > 64KB
                recommendations.append({
                    "type": "large_documents",
                    "message": "Documents are very large. Consider denormalizing or using references.",
                    "avg_size": stats["avg_doc_size"]
                })
            
            # Check for high index size relative to data size
            if stats.get("index_size", 0) > stats.get("storage_size", 0) * 0.5:
                recommendations.append({
                    "type": "high_index_overhead",
                    "message": "Index size is more than 50% of data size. Consider removing unused indexes.",
                    "index_size": stats["index_size"],
                    "data_size": stats["storage_size"]
                })
            
            report["collection_recommendations"][collection_name] = recommendations
        
        # Aggregate all index recommendations
        for query_key, indexes in self.index_recommendations.items():
            for index in indexes:
                collection_name = self.query_stats.get(query_key, {}).get("collection")
                if collection_name:
                    report["index_recommendations"].append({
                        "collection": collection_name,
                        "keys": index["keys"],
                        "query_key": query_key,
                        "reason": index["reason"]
                    })
        
        return report
    
    def _get_explain_plan(self, collection: Collection, query: Dict[str, Any],
                         projection: Optional[Dict[str, Any]] = None,
                         sort: Optional[List[Tuple[str, int]]] = None) -> Dict[str, Any]:
        """
        Get the explain plan for a query.
        
        Args:
            collection: MongoDB collection
            query: Query filter
            projection: Query projection
            sort: Query sort
            
        Returns:
            Dictionary with explain plan
        """
        try:
            # Create query cursor
            cursor = collection.find(query, projection=projection)
            if sort:
                cursor = cursor.sort(sort)
            
            # Get explain plan
            explain_plan = cursor.explain()
            return explain_plan
            
        except Exception as e:
            logger.error(f"Error getting explain plan: {e}")
            return {}
    
    def _check_needs_index(self, explain_plan: Dict[str, Any]) -> bool:
        """
        Check if a query could benefit from an index.
        
        Args:
            explain_plan: MongoDB explain plan
            
        Returns:
            Boolean indicating whether an index is needed
        """
        # Check for collection scan
        if "executionStats" in explain_plan:
            execution_stats = explain_plan["executionStats"]
            
            # If execution stage is COLLSCAN, an index might help
            if "executionStages" in execution_stats:
                stage = execution_stats["executionStages"]
                if stage.get("stage") == "COLLSCAN":
                    # If the collection has many documents, an index would help
                    if execution_stats.get("totalDocsExamined", 0) > 100:
                        return True
                    
                    # If the query is taking a long time, an index would help
                    if execution_stats.get("executionTimeMillis", 0) > 50:
                        return True
        
        # Check for queryPlanner to see if an index could help
        if "queryPlanner" in explain_plan:
            winning_plan = explain_plan["queryPlanner"].get("winningPlan", {})
            
            # Recursive function to check for COLLSCAN
            def has_collscan(plan):
                if plan.get("stage") == "COLLSCAN":
                    return True
                
                # Check child stages
                if "inputStage" in plan:
                    return has_collscan(plan["inputStage"])
                    
                if "inputStages" in plan:
                    return any(has_collscan(stage) for stage in plan["inputStages"])
                    
                return False
            
            if has_collscan(winning_plan):
                return True
        
        return False
    
    def _recommend_indexes(self, collection: Collection, query: Dict[str, Any],
                         sort: Optional[List[Tuple[str, int]]] = None) -> List[Dict[str, Any]]:
        """
        Recommend indexes for a query.
        
        Args:
            collection: MongoDB collection
            query: Query filter
            sort: Query sort
            
        Returns:
            List of recommended indexes
        """
        recommended_indexes = []
        
        # Analyze query for potential index fields
        index_fields = self._extract_index_fields(query)
        
        # If there are fields in the sort, add them to the index
        if sort:
            for field, direction in sort:
                # Add field to index if not already included
                if not any(field == f[0] for f in index_fields):
                    index_fields.append((field, direction))
        
        # Create index recommendation if we have fields
        if index_fields:
            index_name = "_".join([f"{field}_{1 if direction == 1 else -1}" 
                                 for field, direction in index_fields])
            
            recommended_indexes.append({
                "keys": index_fields,
                "name": f"optimization_{index_name}",
                "reason": "Would convert COLLSCAN to IXSCAN for this query"
            })
        
        # If the query has an equality filter followed by a range filter,
        # consider a compound index
        equality_fields = []
        range_fields = []
        
        for field, operators in self._extract_operators(query).items():
            if all(op in ["$eq", None] for op in operators):
                equality_fields.append(field)
            elif any(op in ["$gt", "$gte", "$lt", "$lte"] for op in operators):
                range_fields.append(field)
        
        if equality_fields and range_fields:
            # Create a compound index with equality fields first, then range fields
            compound_fields = [(field, 1) for field in equality_fields]
            for field in range_fields:
                if not any(field == f[0] for f in compound_fields):
                    compound_fields.append((field, 1))
            
            if compound_fields and compound_fields != index_fields:
                index_name = "_".join([f"{field}_{1 if direction == 1 else -1}" 
                                     for field, direction in compound_fields])
                
                recommended_indexes.append({
                    "keys": compound_fields,
                    "name": f"compound_{index_name}",
                    "reason": "Optimized compound index for equality + range query"
                })
        
        return recommended_indexes
    
    def _extract_index_fields(self, query: Dict[str, Any]) -> List[Tuple[str, int]]:
        """
        Extract fields from a query that would benefit from indexing.
        
        Args:
            query: Query filter
            
        Returns:
            List of (field, direction) tuples
        """
        index_fields = []
        
        # Helper function to process nested filters
        def process_filter(filter_dict, parent_path=""):
            for field, value in filter_dict.items():
                # Skip MongoDB operators at the top level
                if field.startswith("$") and not parent_path:
                    if field in ["$and", "$or", "$nor"]:
                        for sub_filter in value:
                            process_filter(sub_filter, parent_path)
                    continue
                
                # Build the full field path
                full_path = f"{parent_path}.{field}" if parent_path else field
                
                # Check if the field is a filter condition
                if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                    # This is a filter with operators, add to index fields
                    if full_path not in [f for f, _ in index_fields]:
                        # Determine direction based on operators
                        # Use ascending (1) by default
                        direction = 1
                        
                        # But if we have a sort-related operator, adjust direction
                        if "$gt" in value or "$gte" in value:
                            direction = 1
                        elif "$lt" in value or "$lte" in value:
                            direction = -1
                            
                        index_fields.append((full_path, direction))
                elif isinstance(value, dict):
                    # This is a nested document, recurse
                    process_filter(value, full_path)
                else:
                    # This is a simple equality condition, add to index fields
                    if full_path not in [f for f, _ in index_fields]:
                        index_fields.append((full_path, 1))
        
        # Process the query
        process_filter(query)
        
        return index_fields
    
    def _extract_operators(self, query: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract operators used for each field in a query.
        
        Args:
            query: Query filter
            
        Returns:
            Dictionary mapping fields to lists of operators
        """
        operators = {}
        
        # Helper function to process nested filters
        def process_filter(filter_dict, parent_path=""):
            for field, value in filter_dict.items():
                # Handle top-level logical operators
                if field in ["$and", "$or", "$nor"]:
                    for sub_filter in value:
                        process_filter(sub_filter, parent_path)
                    continue
                
                # Skip other top-level operators
                if field.startswith("$") and not parent_path:
                    continue
                
                # Build the full field path
                full_path = f"{parent_path}.{field}" if parent_path else field
                
                # Check the value
                if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                    # This is a filter with operators
                    if full_path not in operators:
                        operators[full_path] = []
                    
                    # Add all operators for this field
                    for op in value.keys():
                        if op.startswith("$") and op not in operators[full_path]:
                            operators[full_path].append(op)
                elif isinstance(value, dict):
                    # This is a nested document, recurse
                    process_filter(value, full_path)
                else:
                    # This is a simple equality condition
                    if full_path not in operators:
                        operators[full_path] = []
                    
                    # Equality is represented by None or $eq
                    if None not in operators[full_path]:
                        operators[full_path].append(None)
        
        # Process the query
        process_filter(query)
        
        return operators
    
    def _optimize_query_filter(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a query filter for better performance.
        
        Args:
            query: Query filter
            
        Returns:
            Optimized query filter
        """
        optimized = query.copy()
        
        # Helper function to optimize nested filters
        def optimize_filter(filter_dict):
            result = {}
            
            for field, value in filter_dict.items():
                # Process logical operators
                if field in ["$and", "$or", "$nor"]:
                    result[field] = [optimize_filter(sub_filter) for sub_filter in value]
                    continue
                
                # Process nested documents
                if isinstance(value, dict) and not any(k.startswith("$") for k in value.keys()):
                    result[field] = optimize_filter(value)
                    continue
                
                # Process operators
                if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                    # Check for redundant operators
                    if "$eq" in value and len(value) == 1:
                        # Convert $eq to direct equality
                        result[field] = value["$eq"]
                    else:
                        result[field] = value
                    continue
                
                # Pass through other values
                result[field] = value
            
            return result
        
        return optimize_filter(optimized)
    
    def _optimize_projection(self, projection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a query projection for better performance.
        
        Args:
            projection: Query projection
            
        Returns:
            Optimized query projection
        """
        # Check projection type (inclusion or exclusion)
        has_inclusion = any(v == 1 for v in projection.values())
        has_exclusion = any(v == 0 for v in projection.values())
        
        # Mixed inclusion and exclusion is not allowed except for _id
        if has_inclusion and has_exclusion:
            # Keep only the inclusion fields and _id exclusion if present
            optimized = {
                field: value for field, value in projection.items() 
                if value == 1 or (field == "_id" and value == 0)
            }
        else:
            optimized = projection.copy()
        
        return optimized
    
    def _check_covered_query_potential(self, collection: Collection, 
                                      query: Dict[str, Any],
                                      projection: Optional[Dict[str, Any]] = None,
                                      sort: Optional[List[Tuple[str, int]]] = None) -> bool:
        """
        Check if a query could potentially be covered by an index.
        
        Args:
            collection: MongoDB collection
            query: Query filter
            projection: Query projection
            sort: Query sort
            
        Returns:
            Boolean indicating whether the query could be covered
        """
        # A query can be covered if:
        # 1. It has a projection (inclusion only, not exclusion)
        # 2. All fields in the query, projection, and sort are present in an index
        
        if not projection:
            return False
            
        # Check if projection is inclusion-only
        if not all(v == 1 or (k == "_id" and v == 0) for k, v in projection.items()):
            return False
        
        # Get all fields in query, projection, and sort
        query_fields = set(self._extract_index_fields(query))
        projection_fields = set(k for k, v in projection.items() if v == 1)
        sort_fields = set() if not sort else set(field for field, _ in sort)
        
        # All needed fields
        all_fields = query_fields.union(projection_fields).union(sort_fields)
        
        # Check existing indexes to see if any covers all fields
        try:
            indexes = collection.index_information()
            
            for index_name, index_info in indexes.items():
                if index_name == "_id_":  # Skip _id index
                    continue
                    
                index_fields = set(field for field, _ in index_info["key"])
                if all_fields.issubset(index_fields):
                    return True
        except Exception as e:
            logger.error(f"Error checking indexes: {e}")
        
        return False
    
    def _get_query_key(self, collection_name: str, query: Dict[str, Any],
                     projection: Optional[Dict[str, Any]] = None,
                     sort: Optional[List[Tuple[str, int]]] = None) -> str:
        """
        Generate a unique key for a query.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            projection: Query projection
            sort: Query sort
            
        Returns:
            String key for the query
        """
        # Create a normalized representation of the query
        normalized = {
            "collection": collection_name,
            "query": self._normalize_for_key(query),
            "projection": self._normalize_for_key(projection) if projection else None,
            "sort": sort
        }
        
        # Convert to JSON and hash
        normalized_json = json.dumps(normalized, sort_keys=True)
        return f"collection:{collection_name}:query:{hash(normalized_json)}"
    
    def _normalize_for_key(self, obj: Any) -> Any:
        """
        Normalize an object for use in a cache key.
        
        Args:
            obj: Object to normalize
            
        Returns:
            Normalized object suitable for hashing
        """
        if isinstance(obj, dict):
            return {k: self._normalize_for_key(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [self._normalize_for_key(item) for item in obj]
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj
    
    def _stringify_for_storage(self, obj: Any) -> Any:
        """
        Convert an object to a string representation for storage.
        
        Args:
            obj: Object to stringify
            
        Returns:
            String representation of the object
        """
        try:
            return json.loads(json_util.dumps(obj))
        except (TypeError, ValueError):
            return str(obj)


# Decorator for query optimization
def optimized_query(explain: bool = False, apply_indexes: bool = False):
    """
    Decorator for optimizing MongoDB queries.
    
    Args:
        explain: Whether to return query explanation
        apply_indexes: Whether to apply recommended indexes
        
    Returns:
        Decorated function with query optimization
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract collection from args
            collection = None
            for arg in args:
                if isinstance(arg, Collection):
                    collection = arg
                    break
            
            if not collection:
                # Try to find in kwargs
                for arg_name, arg_value in kwargs.items():
                    if isinstance(arg_value, Collection):
                        collection = arg_value
                        break
            
            if not collection:
                # Can't optimize without a collection
                return func(*args, **kwargs)
            
            # Extract query, projection, and sort from args and kwargs
            query = None
            projection = None
            sort = None
            
            # Inspect function signature to identify parameters
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Try to extract query, projection, and sort from args and kwargs
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    if param_name == "filter" or param_name == "query":
                        query = arg
                    elif param_name == "projection":
                        projection = arg
                    elif param_name == "sort":
                        sort = arg
            
            # Check kwargs
            query = kwargs.get("filter", kwargs.get("query", query))
            projection = kwargs.get("projection", projection)
            sort = kwargs.get("sort", sort)
            
            # If we couldn't identify the query, just call the original function
            if query is None:
                return func(*args, **kwargs)
            
            # Get the optimizer
            optimizer = QueryOptimizer(collection.database.client)
            
            # Optimize the query
            optimization_result = optimizer.optimize_query(
                collection, query, projection, sort
            )
            
            # Update args or kwargs with optimized values
            new_args = list(args)
            new_kwargs = kwargs.copy()
            
            # Update args
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    if param_name == "filter" or param_name == "query":
                        new_args[i] = optimization_result["optimized_query"]
                    elif param_name == "projection" and optimization_result["optimized_projection"]:
                        new_args[i] = optimization_result["optimized_projection"]
                    elif param_name == "sort" and optimization_result["optimized_sort"]:
                        new_args[i] = optimization_result["optimized_sort"]
            
            # Update kwargs
            if "filter" in kwargs:
                new_kwargs["filter"] = optimization_result["optimized_query"]
            elif "query" in kwargs:
                new_kwargs["query"] = optimization_result["optimized_query"]
                
            if "projection" in kwargs and optimization_result["optimized_projection"]:
                new_kwargs["projection"] = optimization_result["optimized_projection"]
                
            if "sort" in kwargs and optimization_result["optimized_sort"]:
                new_kwargs["sort"] = optimization_result["optimized_sort"]
            
            # Apply indexes if requested
            if apply_indexes:
                # Analyze the query to generate index recommendations
                optimizer.analyze_query(collection, optimization_result["optimized_query"],
                                       optimization_result["optimized_projection"],
                                       optimization_result["optimized_sort"])
                
                # Apply recommended indexes
                optimizer.apply_recommended_indexes(collection)
            
            # Call the original function with optimized parameters
            result = func(*new_args, **new_kwargs)
            
            # Generate explanation if requested
            if explain and isinstance(result, Cursor):
                # Execute the cursor to ensure the query runs
                explanation = {
                    "optimization": optimization_result,
                    "explain_plan": result.explain()
                }
                
                # If the cursor has been exhausted, recreate it
                result = func(*new_args, **new_kwargs)
                
                # Store explanation with the result
                if hasattr(result, "__dict__"):
                    result.__dict__["_explanation"] = explanation
                
            return result
        
        return wrapper
    
    return decorator

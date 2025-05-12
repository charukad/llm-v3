"""
Advanced caching system for mathematical computations.
Provides intelligent caching strategies for optimizing performance.
"""
import time
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Callable
import redis
import pickle
import sympy as sp
from functools import wraps
import inspect
import threading
from math_llm_system.orchestration.monitoring.logger import get_logger

logger = get_logger("math_processing.computation_cache")

class ComputationCache:
    """
    Provides caching for expensive mathematical computations.
    Implements multiple caching strategies and automatic invalidation.
    """
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ComputationCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 default_ttl: int = 3600, 
                 max_local_cache_size: int = 1000):
        """
        Initialize the computation cache.
        
        Args:
            redis_url: URL for Redis connection
            default_ttl: Default TTL for cached items in seconds
            max_local_cache_size: Maximum size of local in-memory cache
        """
        # Only initialize once for singleton
        if self._initialized:
            return
            
        self.default_ttl = default_ttl
        self.max_local_cache_size = max_local_cache_size
        
        # In-memory LRU cache for fast access to frequent items
        self.local_cache = {}
        self.local_cache_access_times = {}
        
        # Connect to Redis for distributed caching
        try:
            self.redis = redis.from_url(redis_url)
            self.redis_available = True
            logger.info(f"Connected to Redis cache at {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using local cache only.")
            self.redis_available = False
        
        # Cache hit/miss metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0,
            "stores": 0,
            "invalidations": 0
        }
        
        # Set of dependency keys to track computation dependencies
        self.dependency_graph = {}
        
        self._initialized = True
        logger.info("Computation cache initialized")
    
    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """
        Generate a cache key from function name and arguments.
        
        Args:
            func_name: Name of the function being cached
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            String cache key
        """
        # Convert args and kwargs to a JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, sp.Basic):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        serializable_args = [make_serializable(arg) for arg in args]
        serializable_kwargs = {k: make_serializable(v) for k, v in kwargs.items()}
        
        # Create a string representation of the function call
        key_data = {
            "func": func_name,
            "args": serializable_args,
            "kwargs": serializable_kwargs
        }
        
        # Generate a hash of the key data
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()
        
        return f"math:computation:{key_hash}"
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (hit, value) where hit is True if the key was found
        """
        # First check local cache for fastest access
        if key in self.local_cache:
            self.metrics["hits"] += 1
            self.metrics["local_hits"] += 1
            self.local_cache_access_times[key] = time.time()
            logger.debug(f"Local cache hit for key: {key}")
            return True, self.local_cache[key]
        
        # If not in local cache, check Redis
        if self.redis_available:
            try:
                cached_value = self.redis.get(key)
                if cached_value is not None:
                    # Deserialize the value
                    value = pickle.loads(cached_value)
                    
                    # Add to local cache for faster future access
                    self._add_to_local_cache(key, value)
                    
                    self.metrics["hits"] += 1
                    self.metrics["redis_hits"] += 1
                    logger.debug(f"Redis cache hit for key: {key}")
                    return True, value
            except Exception as e:
                logger.error(f"Error retrieving from Redis cache: {e}")
        
        # Cache miss
        self.metrics["misses"] += 1
        return False, None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            dependencies: Optional[List[str]] = None) -> bool:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default_ttl if None)
            dependencies: List of keys this computation depends on
            
        Returns:
            Boolean indicating success
        """
        ttl = ttl if ttl is not None else self.default_ttl
        
        # Store in local cache
        self._add_to_local_cache(key, value)
        
        # Track dependencies for invalidation
        if dependencies:
            for dep_key in dependencies:
                if dep_key not in self.dependency_graph:
                    self.dependency_graph[dep_key] = set()
                self.dependency_graph[dep_key].add(key)
        
        # Store in Redis for distributed cache
        if self.redis_available:
            try:
                # Serialize the value
                serialized_value = pickle.dumps(value)
                
                # Set in Redis with TTL
                self.redis.setex(key, ttl, serialized_value)
            except Exception as e:
                logger.error(f"Error storing in Redis cache: {e}")
                return False
        
        self.metrics["stores"] += 1
        return True
    
    def invalidate(self, key: str, recursive: bool = True) -> int:
        """
        Invalidate a cache entry and optionally its dependents.
        
        Args:
            key: Cache key to invalidate
            recursive: Whether to recursively invalidate dependent computations
            
        Returns:
            Number of invalidated keys
        """
        invalidated = 0
        
        # Track dependents to invalidate
        to_invalidate = set([key])
        
        # Recursively find all dependent keys if requested
        if recursive and key in self.dependency_graph:
            dependents = set(self.dependency_graph[key])
            while dependents:
                dependent = dependents.pop()
                to_invalidate.add(dependent)
                if dependent in self.dependency_graph:
                    dependents.update(self.dependency_graph[dependent])
        
        # Invalidate all identified keys
        for invalid_key in to_invalidate:
            # Remove from local cache
            if invalid_key in self.local_cache:
                del self.local_cache[invalid_key]
                if invalid_key in self.local_cache_access_times:
                    del self.local_cache_access_times[invalid_key]
                invalidated += 1
            
            # Remove from Redis
            if self.redis_available:
                try:
                    self.redis.delete(invalid_key)
                    invalidated += 1
                except Exception as e:
                    logger.error(f"Error invalidating Redis cache: {e}")
        
        self.metrics["invalidations"] += invalidated
        return invalidated
    
    def _add_to_local_cache(self, key: str, value: Any):
        """
        Add a value to the local cache, managing size limits.
        
        Args:
            key: Cache key
            value: Value to store
        """
        # If we've reached the max size, evict least recently used item
        if len(self.local_cache) >= self.max_local_cache_size:
            self._evict_lru()
        
        # Add to local cache
        self.local_cache[key] = value
        self.local_cache_access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict the least recently used item from the local cache."""
        if not self.local_cache_access_times:
            return
            
        # Find the least recently used key
        lru_key = min(self.local_cache_access_times.items(), key=lambda x: x[1])[0]
        
        # Remove from local cache
        del self.local_cache[lru_key]
        del self.local_cache_access_times[lru_key]
    
    def clear(self):
        """Clear all cached values."""
        # Clear local cache
        self.local_cache.clear()
        self.local_cache_access_times.clear()
        self.dependency_graph.clear()
        
        # Clear Redis keys (with pattern matching)
        if self.redis_available:
            try:
                keys = self.redis.keys("math:computation:*")
                if keys:
                    self.redis.delete(*keys)
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary of cache metrics
        """
        total_requests = self.metrics["hits"] + self.metrics["misses"]
        hit_ratio = self.metrics["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "hits": self.metrics["hits"],
            "misses": self.metrics["misses"],
            "stores": self.metrics["stores"],
            "invalidations": self.metrics["invalidations"],
            "hit_ratio": hit_ratio,
            "local_cache_size": len(self.local_cache),
            "local_hit_ratio": self.metrics["local_hits"] / self.metrics["hits"] if self.metrics["hits"] > 0 else 0
        }
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get health information about the cache.
        
        Returns:
            Dictionary with health information
        """
        return {
            "local_cache_size": len(self.local_cache),
            "dependency_graph_size": len(self.dependency_graph),
            "redis_available": self.redis_available,
            "redis_ping": self._ping_redis() if self.redis_available else False
        }
    
    def _ping_redis(self) -> bool:
        """
        Check if Redis is responsive.
        
        Returns:
            Boolean indicating Redis responsiveness
        """
        if not self.redis_available:
            return False
            
        try:
            return self.redis.ping()
        except Exception:
            return False


# Decorator for caching mathematical operations
def cached_computation(ttl: Optional[int] = None, 
                       dynamic_ttl: Optional[Callable[[Any], int]] = None,
                       track_dependencies: bool = True):
    """
    Decorator for caching mathematical computation results.
    
    Args:
        ttl: Time-to-live for cache entries in seconds
        dynamic_ttl: Function to calculate TTL based on result
        track_dependencies: Whether to track dependencies for invalidation
        
    Returns:
        Decorated function with caching
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the cache instance
            cache = ComputationCache()
            
            # Generate cache key
            func_name = f"{func.__module__}.{func.__name__}"
            key = cache._generate_key(func_name, args, kwargs)
            
            # Check if result is cached
            hit, cached_result = cache.get(key)
            if hit:
                return cached_result
            
            # Not cached, compute the result
            result = func(*args, **kwargs)
            
            # Determine TTL
            if dynamic_ttl is not None:
                result_ttl = dynamic_ttl(result)
            else:
                result_ttl = ttl
            
            # Track dependencies if enabled
            dependencies = None
            if track_dependencies:
                # Inspect the function signature to identify parameters that
                # might be used as keys in other cached operations
                sig = inspect.signature(func)
                dependencies = []
                
                # Extract potential dependencies from arguments
                for param_name, param in sig.parameters.items():
                    if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                        index = list(sig.parameters.keys()).index(param_name)
                        if index < len(args):
                            arg_value = args[index]
                            if isinstance(arg_value, str) and arg_value.startswith("math:computation:"):
                                dependencies.append(arg_value)
                
                # Extract potential dependencies from keyword arguments
                for key, value in kwargs.items():
                    if isinstance(value, str) and value.startswith("math:computation:"):
                        dependencies.append(value)
            
            # Cache the result
            cache.set(key, result, ttl=result_ttl, dependencies=dependencies)
            
            return result
        
        return wrapper
    
    return decorator

"""
Performance Optimizer Core Implementation

This module provides the central performance optimization engine that:
1. Coordinates all optimization components
2. Dynamically adjusts optimization levels based on system load
3. Detects performance bottlenecks automatically
4. Provides optimization recommendations
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any

from orchestration.performance.resource_manager import ResourceManager
from math_processing.computation.computation_cache import ComputationCache
from math_processing.computation.parallel_processor import ParallelProcessor
from database.optimization.query_optimizer import QueryOptimizer
from api.rest.middlewares.request_batcher import RequestBatcher

# Configure logger
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Core performance optimization engine that integrates all optimization components
    and provides dynamic optimization based on system load and usage patterns.
    """
    
    # Optimization level constants
    OPTIMIZATION_LEVEL_MINIMAL = 0
    OPTIMIZATION_LEVEL_BALANCED = 1
    OPTIMIZATION_LEVEL_AGGRESSIVE = 2
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance optimizer with all optimization components.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        
        # Initialize component instances
        self.resource_manager = ResourceManager()
        self.computation_cache = ComputationCache()
        self.parallel_processor = ParallelProcessor()
        self.query_optimizer = QueryOptimizer()
        self.request_batcher = RequestBatcher()
        
        # Performance metrics
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "response_times": [],
            "cache_hit_ratio": [],
            "database_query_times": [],
            "optimization_level_history": [],
        }
        
        # Current optimization level
        self.current_optimization_level = self.OPTIMIZATION_LEVEL_BALANCED
        
        # Bottleneck detection thresholds
        self.thresholds = {
            "high_cpu": self.config.get("high_cpu_threshold", 80.0),  # percentage
            "high_memory": self.config.get("high_memory_threshold", 80.0),  # percentage
            "slow_response": self.config.get("slow_response_threshold", 2.0),  # seconds
            "low_cache_hit": self.config.get("low_cache_hit_threshold", 0.4),  # ratio
            "slow_query": self.config.get("slow_query_threshold", 0.5),  # seconds
        }
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Performance optimizer initialized with balanced optimization level")

    def optimize_request_handling(self, request_type: str, payload_size: int) -> Dict[str, Any]:
        """
        Optimize request handling based on request type and payload size.
        
        Args:
            request_type: Type of the request (e.g., 'computation', 'ocr', 'visualization')
            payload_size: Size of the request payload in bytes
            
        Returns:
            Dictionary with optimization settings for this request
        """
        optimization_settings = {
            "use_cache": True,
            "parallel_processing": False,
            "batching_enabled": False,
            "timeout": 30.0,  # seconds
            "max_resources": {"cpu_percent": 50, "memory_mb": 1024},
        }
        
        # Adjust based on current optimization level
        if self.current_optimization_level == self.OPTIMIZATION_LEVEL_AGGRESSIVE:
            optimization_settings["parallel_processing"] = True
            optimization_settings["batching_enabled"] = True
            optimization_settings["timeout"] = 60.0  # Allow more time for batching
            optimization_settings["max_resources"]["cpu_percent"] = 80
            optimization_settings["max_resources"]["memory_mb"] = 2048
        
        # Adjust based on request type
        if request_type == "computation":
            # Mathematical computations benefit from parallel processing
            optimization_settings["parallel_processing"] = True
            
        elif request_type == "visualization":
            # Visualizations can be resource-intensive
            optimization_settings["max_resources"]["memory_mb"] = 1536
            
        elif request_type == "ocr":
            # OCR requests need more time but less caching
            optimization_settings["timeout"] = 45.0
            optimization_settings["use_cache"] = False
        
        # Adjust based on payload size
        if payload_size > 1024 * 1024:  # More than 1MB
            # Large payloads need more resources
            optimization_settings["max_resources"]["memory_mb"] *= 1.5
            optimization_settings["timeout"] *= 1.5
        
        return optimization_settings
    
    def optimize_computation(self, expression_type: str, complexity: int) -> Dict[str, Any]:
        """
        Optimize mathematical computation based on expression type and complexity.
        
        Args:
            expression_type: Type of mathematical expression (e.g., 'algebra', 'calculus')
            complexity: Estimated complexity of the computation (1-10)
            
        Returns:
            Dictionary with optimization settings for this computation
        """
        optimization_settings = {
            "use_cache": True,
            "parallel_processing": complexity > 5,
            "max_processors": max(1, min(8, complexity // 2)),
            "optimization_techniques": ["simplify_first"],
        }
        
        # Adjust based on expression type
        if expression_type == "calculus":
            optimization_settings["optimization_techniques"].append("symbolic_first")
            optimization_settings["parallel_processing"] = complexity > 3
            
        elif expression_type == "linear_algebra":
            optimization_settings["optimization_techniques"].append("matrix_optimization")
            optimization_settings["parallel_processing"] = True
            
        elif expression_type == "statistics":
            optimization_settings["optimization_techniques"].append("numerical_approximation")
        
        # Adjust based on current system load
        current_load = self.resource_manager.get_system_load()
        if current_load["cpu_percent"] > 70:
            # Reduce parallelism under high CPU load
            optimization_settings["max_processors"] = max(1, optimization_settings["max_processors"] // 2)
        
        if current_load["memory_percent"] > 80:
            # Add memory-saving technique under high memory load
            optimization_settings["optimization_techniques"].append("memory_efficient")
        
        return optimization_settings
    
    def optimize_database_query(self, query_type: str, estimated_result_size: int) -> Dict[str, Any]:
        """
        Optimize database query based on query type and estimated result size.
        
        Args:
            query_type: Type of database query (e.g., 'find', 'aggregate')
            estimated_result_size: Estimated size of the query result (number of documents)
            
        Returns:
            Dictionary with optimization settings for this query
        """
        optimization_settings = {
            "use_index": True,
            "batch_size": 100,
            "projection": None,  # No field limitation by default
            "max_time_ms": 10000,  # 10 seconds
        }
        
        # Adjust based on query type
        if query_type == "aggregate":
            optimization_settings["batch_size"] = 50
            optimization_settings["max_time_ms"] = 30000  # 30 seconds for aggregations
            
        elif query_type == "text_search":
            optimization_settings["max_time_ms"] = 20000  # 20 seconds for text search
        
        # Adjust based on estimated result size
        if estimated_result_size > 1000:
            optimization_settings["batch_size"] = 250
            # For large result sets, consider limiting fields
            optimization_settings["projection"] = {"_id": 1, "essential_field": 1}
        
        # Apply query optimization suggestions from the query optimizer
        query_suggestions = self.query_optimizer.get_optimization_suggestions(query_type)
        if query_suggestions:
            optimization_settings.update(query_suggestions)
        
        return optimization_settings
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks in the system.
        
        Returns:
            List of detected bottlenecks with details
        """
        bottlenecks = []
        current_metrics = self._get_current_metrics()
        
        # Check CPU usage
        if current_metrics["cpu_usage"] > self.thresholds["high_cpu"]:
            bottlenecks.append({
                "type": "high_cpu_usage",
                "value": current_metrics["cpu_usage"],
                "threshold": self.thresholds["high_cpu"],
                "recommendation": "Consider scaling horizontally or reducing parallel processing"
            })
        
        # Check memory usage
        if current_metrics["memory_usage"] > self.thresholds["high_memory"]:
            bottlenecks.append({
                "type": "high_memory_usage",
                "value": current_metrics["memory_usage"],
                "threshold": self.thresholds["high_memory"],
                "recommendation": "Consider increasing memory or optimizing memory-intensive operations"
            })
        
        # Check response times
        if current_metrics["avg_response_time"] > self.thresholds["slow_response"]:
            bottlenecks.append({
                "type": "slow_response_time",
                "value": current_metrics["avg_response_time"],
                "threshold": self.thresholds["slow_response"],
                "recommendation": "Investigate slow components or increase caching"
            })
        
        # Check cache hit ratio
        if current_metrics["cache_hit_ratio"] < self.thresholds["low_cache_hit"]:
            bottlenecks.append({
                "type": "low_cache_hit_ratio",
                "value": current_metrics["cache_hit_ratio"],
                "threshold": self.thresholds["low_cache_hit"],
                "recommendation": "Adjust cache size or caching strategy"
            })
        
        # Check database query times
        if current_metrics["avg_query_time"] > self.thresholds["slow_query"]:
            bottlenecks.append({
                "type": "slow_database_queries",
                "value": current_metrics["avg_query_time"],
                "threshold": self.thresholds["slow_query"],
                "recommendation": "Review index coverage or query patterns"
            })
        
        return bottlenecks
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on system metrics.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        bottlenecks = self.detect_bottlenecks()
        
        # Process detected bottlenecks into actionable recommendations
        for bottleneck in bottlenecks:
            recommendations.append({
                "issue": bottleneck["type"],
                "recommendation": bottleneck["recommendation"],
                "priority": "high" if bottleneck["value"] / bottleneck["threshold"] > 1.5 else "medium"
            })
        
        # Add general optimization recommendations based on metrics
        current_metrics = self._get_current_metrics()
        
        # Check for caching opportunities
        if current_metrics["cache_hit_ratio"] < 0.6:  # Room for improvement
            recommendations.append({
                "issue": "suboptimal_caching",
                "recommendation": "Increase cache size or adjust caching strategy for frequently accessed data",
                "priority": "medium"
            })
        
        # Check for parallelization opportunities
        if current_metrics["cpu_usage"] < 50 and current_metrics["avg_response_time"] > 1.0:
            recommendations.append({
                "issue": "underutilized_parallelization",
                "recommendation": "Increase parallel processing for computational tasks",
                "priority": "medium"
            })
        
        # Check for database optimizations
        if len(current_metrics["slow_queries"]) > 0:
            recommendations.append({
                "issue": "slow_database_queries",
                "recommendation": f"Optimize the following slow queries: {', '.join(current_metrics['slow_queries'])}",
                "priority": "high"
            })
        
        return recommendations
    
    def adjust_optimization_level(self) -> None:
        """
        Dynamically adjust the optimization level based on system metrics.
        """
        current_metrics = self._get_current_metrics()
        
        # Decision logic for optimization level
        if (current_metrics["cpu_usage"] > 90 or 
            current_metrics["memory_usage"] > 90 or
            current_metrics["avg_response_time"] > 5.0):
            # System under heavy load, apply aggressive optimization
            new_level = self.OPTIMIZATION_LEVEL_AGGRESSIVE
            
        elif (current_metrics["cpu_usage"] < 30 and
              current_metrics["memory_usage"] < 30 and
              current_metrics["avg_response_time"] < 0.5):
            # System lightly loaded, minimal optimization is sufficient
            new_level = self.OPTIMIZATION_LEVEL_MINIMAL
            
        else:
            # Balanced load, use balanced optimization
            new_level = self.OPTIMIZATION_LEVEL_BALANCED
        
        # Apply the new optimization level if it changed
        if new_level != self.current_optimization_level:
            logger.info(f"Adjusting optimization level from {self.current_optimization_level} to {new_level}")
            self.current_optimization_level = new_level
            self.metrics["optimization_level_history"].append((time.time(), new_level))
            
            # Apply changes to components based on new level
            self._apply_optimization_level()
    
    def _apply_optimization_level(self) -> None:
        """
        Apply the current optimization level to all components.
        """
        # Apply to resource manager
        if self.current_optimization_level == self.OPTIMIZATION_LEVEL_MINIMAL:
            self.resource_manager.set_resource_limits(cpu_percent=30, memory_percent=40)
        elif self.current_optimization_level == self.OPTIMIZATION_LEVEL_BALANCED:
            self.resource_manager.set_resource_limits(cpu_percent=60, memory_percent=70)
        else:  # AGGRESSIVE
            self.resource_manager.set_resource_limits(cpu_percent=90, memory_percent=90)
        
        # Apply to computation cache
        if self.current_optimization_level == self.OPTIMIZATION_LEVEL_MINIMAL:
            self.computation_cache.set_cache_size(100)  # Small cache
        elif self.current_optimization_level == self.OPTIMIZATION_LEVEL_BALANCED:
            self.computation_cache.set_cache_size(1000)  # Medium cache
        else:  # AGGRESSIVE
            self.computation_cache.set_cache_size(5000)  # Large cache
        
        # Apply to parallel processor
        if self.current_optimization_level == self.OPTIMIZATION_LEVEL_MINIMAL:
            self.parallel_processor.set_parallelism(2)  # Minimal parallelism
        elif self.current_optimization_level == self.OPTIMIZATION_LEVEL_BALANCED:
            self.parallel_processor.set_parallelism(4)  # Balanced parallelism
        else:  # AGGRESSIVE
            self.parallel_processor.set_parallelism(8)  # Maximum parallelism
        
        # Apply to request batcher
        if self.current_optimization_level == self.OPTIMIZATION_LEVEL_MINIMAL:
            self.request_batcher.set_batching_parameters(enabled=False)
        elif self.current_optimization_level == self.OPTIMIZATION_LEVEL_BALANCED:
            self.request_batcher.set_batching_parameters(enabled=True, max_batch_size=5, max_wait_time=0.1)
        else:  # AGGRESSIVE
            self.request_batcher.set_batching_parameters(enabled=True, max_batch_size=20, max_wait_time=0.5)
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics from all components.
        
        Returns:
            Dictionary with current metrics
        """
        # Get system resources
        system_load = self.resource_manager.get_system_load()
        
        # Get cache metrics
        cache_metrics = self.computation_cache.get_metrics()
        
        # Get database metrics
        db_metrics = self.query_optimizer.get_metrics()
        
        # Combine all metrics
        current_metrics = {
            "cpu_usage": system_load["cpu_percent"],
            "memory_usage": system_load["memory_percent"],
            "avg_response_time": sum(self.metrics["response_times"][-10:]) / max(1, len(self.metrics["response_times"][-10:])) if self.metrics["response_times"] else 0,
            "cache_hit_ratio": cache_metrics["hit_ratio"],
            "avg_query_time": db_metrics["average_query_time"],
            "slow_queries": db_metrics["slow_queries"],
        }
        
        return current_metrics
    
    def record_metrics(self, metric_type: str, value: Union[float, int]) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_type: Type of metric (e.g., 'response_time', 'cache_hit')
            value: Metric value
        """
        if metric_type == "response_time":
            self.metrics["response_times"].append(value)
            # Keep only the last 1000 response times
            if len(self.metrics["response_times"]) > 1000:
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def _monitoring_loop(self) -> None:
        """
        Background monitoring loop that periodically adjusts optimization levels.
        """
        while self.monitoring_active:
            try:
                # Record current metrics
                system_load = self.resource_manager.get_system_load()
                self.metrics["cpu_usage"].append((time.time(), system_load["cpu_percent"]))
                self.metrics["memory_usage"].append((time.time(), system_load["memory_percent"]))
                
                # Keep metrics history limited to prevent memory bloat
                for metric_list in self.metrics.values():
                    if isinstance(metric_list, list) and len(metric_list) > 1000:
                        metric_list = metric_list[-1000:]
                
                # Adjust optimization level based on current metrics
                self.adjust_optimization_level()
                
                # Generate optimization recommendations periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    recommendations = self.get_optimization_recommendations()
                    if recommendations:
                        logger.info(f"Optimization recommendations: {recommendations}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for monitoring interval
            time.sleep(self.config.get("monitoring_interval", 30))  # Default: check every 30 seconds
    
    def shutdown(self) -> None:
        """
        Shutdown the performance optimizer and its components.
        """
        logger.info("Shutting down performance optimizer")
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # Shutdown components
        self.resource_manager.shutdown()
        self.computation_cache.shutdown()
        self.parallel_processor.shutdown()
        self.query_optimizer.shutdown()
        self.request_batcher.shutdown()

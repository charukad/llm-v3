"""
Unit tests for the performance optimizer.
"""

import unittest
import time
from unittest.mock import MagicMock, patch

from orchestration.performance.performance_optimizer import (
    PerformanceOptimizer,
    ComputationCache,
    ResourceManager,
    ParallelExecutor,
    MessageBatcher,
    OptimizationLevel,
    ResourceAllocation,
    optimize_function,
    get_performance_optimizer
)

class TestComputationCache(unittest.TestCase):
    """Tests for the ComputationCache class."""
    
    def setUp(self):
        self.cache = ComputationCache(max_size_mb=10)
    
    def test_cache_key_generation(self):
        """Test generation of cache keys."""
        inputs1 = {"a": 1, "b": 2}
        inputs2 = {"b": 2, "a": 1}  # Same values, different order
        
        key1 = self.cache.generate_key(inputs1, "test_func")
        key2 = self.cache.generate_key(inputs2, "test_func")
        
        # Keys should be the same despite different order
        self.assertEqual(key1, key2)
        
        # Different function name should yield different key
        key3 = self.cache.generate_key(inputs1, "other_func")
        self.assertNotEqual(key1, key3)
    
    def test_cache_set_get(self):
        """Test setting and getting values from cache."""
        key = "test_key"
        value = {"result": 42}
        
        # Set the value
        self.cache.set(key, value)
        
        # Get the value
        cached_value = self.cache.get(key)
        self.assertEqual(cached_value, value)
        
        # Get a non-existent key
        missing_value = self.cache.get("missing_key")
        self.assertIsNone(missing_value)
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        key = "test_key"
        value = {"result": 42}
        
        # Set the value
        self.cache.set(key, value)
        
        # Invalidate the entry
        result = self.cache.invalidate(key)
        self.assertTrue(result)
        
        # Verify it's gone
        cached_value = self.cache.get(key)
        self.assertIsNone(cached_value)
        
        # Invalidate non-existent key
        result = self.cache.invalidate("missing_key")
        self.assertFalse(result)
    
    def test_cache_size_management(self):
        """Test cache size management and eviction."""
        # Create small cache for testing
        small_cache = ComputationCache(max_size_mb=0.001)  # ~1KB
        
        # Add some entries
        for i in range(10):
            key = f"key_{i}"
            # Create value with known size (roughly 100 bytes each)
            value = {"data": "x" * 100}
            small_cache.set(key, value)
        
        # Check size is managed
        self.assertLessEqual(small_cache.get_current_size_mb(), 0.001)
        
        # Older entries should be evicted (LRU policy)
        self.assertIsNone(small_cache.get("key_0"))
        self.assertIsNotNone(small_cache.get("key_9"))


class TestResourceManager(unittest.TestCase):
    """Tests for the ResourceManager class."""
    
    def setUp(self):
        self.manager = ResourceManager()
    
    def test_allocate_for_computation(self):
        """Test resource allocation for computation."""
        # Test allocation for various computation types
        symbolic_allocation = self.manager.allocate_for_computation(
            "symbolic_computation", {"expression": "x^2 + 3*x + 2"})
        
        matrix_allocation = self.manager.allocate_for_computation(
            "matrix_operations", {"matrix": [[1, 2], [3, 4]]})
        
        # Basic allocation should have reasonable defaults
        self.assertIsInstance(symbolic_allocation, ResourceAllocation)
        self.assertGreater(symbolic_allocation.cpu_cores, 0)
        self.assertGreater(symbolic_allocation.memory_mb, 0)
        
        # Matrix operations should allocate more resources
        self.assertGreaterEqual(matrix_allocation.memory_mb, symbolic_allocation.memory_mb)
    
    def test_get_system_load(self):
        """Test getting system load."""
        load = self.manager.get_system_load()
        
        # Should return a dictionary with certain keys
        self.assertIsInstance(load, dict)
        self.assertIn("cpu", load)
        self.assertIn("memory", load)
        
        # Values should be reasonable
        self.assertGreaterEqual(load["cpu"], 0)
        self.assertLessEqual(load["cpu"], 100)
        self.assertGreaterEqual(load["memory"], 0)
        self.assertLessEqual(load["memory"], 100)


class TestParallelExecutor(unittest.TestCase):
    """Tests for the ParallelExecutor class."""
    
    def setUp(self):
        self.executor = ParallelExecutor(max_workers=2)
    
    def test_execute(self):
        """Test parallel execution of a function."""
        def example_func(x):
            return x * 2
        
        # Execute a simple function
        result = self.executor.execute(example_func, 5)
        self.assertEqual(result, 10)
    
    def test_execute_map(self):
        """Test parallel execution of function on multiple items."""
        def example_func(x):
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        results = self.executor.execute_map(example_func, items)
        
        self.assertEqual(results, [2, 4, 6, 8, 10])
    
    def test_execute_with_resource_allocation(self):
        """Test execution with resource allocation."""
        def example_func(x):
            return x * 2
        
        allocation = ResourceAllocation(cpu_cores=1.5, memory_mb=512)
        result = self.executor.execute(example_func, 5, resource_allocation=allocation)
        
        self.assertEqual(result, 10)


class TestMessageBatcher(unittest.TestCase):
    """Tests for the MessageBatcher class."""
    
    def setUp(self):
        self.batcher = MessageBatcher(batch_size=3, max_wait_time=0.1)
    
    def test_intercept_message(self):
        """Test message interception."""
        # Create a test message
        message = {
            "header": {
                "recipient": "test_agent",
                "priority": "normal",
                "message_type": "test_message"
            },
            "body": {"data": "test"}
        }
        
        # Intercept message
        result = self.batcher.intercept_message(message)
        
        # Should return True (message intercepted)
        self.assertTrue(result)
        
        # Batch should contain the message
        self.assertIn("test_agent", self.batcher.batches)
        self.assertEqual(len(self.batcher.batches["test_agent"]), 1)
    
    def test_urgent_message_not_intercepted(self):
        """Test that urgent messages are not intercepted."""
        # Create an urgent message
        urgent_message = {
            "header": {
                "recipient": "test_agent",
                "priority": "high",
                "message_type": "urgent_message"
            },
            "body": {"data": "urgent"}
        }
        
        # Intercept message
        result = self.batcher.intercept_message(urgent_message)
        
        # Should return False (message not intercepted)
        self.assertFalse(result)
    
    def test_batch_full_sends(self):
        """Test that full batches are sent."""
        # Use a mock to capture sent batches
        self.batcher._send_batch = MagicMock()
        
        # Create test messages
        recipient = "test_agent"
        messages = []
        for i in range(3):  # Batch size is 3
            message = {
                "header": {
                    "recipient": recipient,
                    "priority": "normal",
                    "message_type": "test_message"
                },
                "body": {"data": f"test_{i}"}
            }
            messages.append(message)
            
            # Intercept each message
            self.batcher.intercept_message(message)
        
        # Verify batch was sent
        self.batcher._send_batch.assert_called_once_with(recipient)


class TestPerformanceOptimizer(unittest.TestCase):
    """Tests for the PerformanceOptimizer class."""
    
    def setUp(self):
        self.optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.BASIC,
            max_workers=2,
            cache_size_mb=10,
            message_batch_size=3
        )
    
    def test_optimize_computation_with_cache(self):
        """Test computation optimization with caching."""
        # Create a mock computation function
        computation_func = MagicMock(return_value={"result": 42})
        
        # First call should compute
        inputs = {"a": 1, "b": 2}
        result1 = self.optimizer.optimize_computation(computation_func, inputs)
        
        # Function should be called
        computation_func.assert_called_once_with(inputs)
        self.assertEqual(result1, {"result": 42})
        
        # Reset mock
        computation_func.reset_mock()
        
        # Second call with same inputs should use cache
        result2 = self.optimizer.optimize_computation(computation_func, inputs)
        
        # Function should not be called
        computation_func.assert_not_called()
        self.assertEqual(result2, {"result": 42})
        
        # Cache hit count should be 1
        self.assertEqual(self.optimizer.cache_hits, 1)
    
    def test_basic_optimization_strategy(self):
        """Test basic optimization strategy."""
        # Create a computation that supports parallelization
        def parallel_comp(inputs):
            if "parallel_chunks" in inputs:
                # Process chunks and return
                return sum(inputs["parallel_chunks"])
            return inputs.get("value", 0)
        
        # Create inputs that indicate parallelization
        inputs = {
            "can_parallelize": True,
            "parallel_chunks": [1, 2, 3, 4, 5]
        }
        
        # Apply basic optimization
        result = self.optimizer._apply_basic_optimizations(
            parallel_comp, 
            inputs, 
            ResourceAllocation(cpu_cores=2.0, memory_mb=512)
        )
        
        # Should use parallelization
        self.assertEqual(result, 15)  # Sum of chunks
    
    def test_aggressive_optimization_strategy(self):
        """Test aggressive optimization strategy."""
        # Create a mock computation function
        def symbolic_comp(inputs):
            # Check for optimization flags
            if "optimize_aggressively" in inputs:
                if inputs.get("allow_numerical", False):
                    return {"result": "numerical_approximation", "precise": False}
            return {"result": "exact_symbolic", "precise": True}
        
        # Apply aggressive optimization to a symbolic computation
        result = self.optimizer._apply_aggressive_optimizations(
            symbolic_comp, 
            {"expression": "complex_integral"}, 
            ResourceAllocation(cpu_cores=1.0, memory_mb=512)
        )
        
        # Should use numerical approximation
        self.assertEqual(result["result"], "numerical_approximation")
        self.assertFalse(result["precise"])
    
    def test_optimization_level_selection(self):
        """Test optimization level selection."""
        # Create a mock computation function
        computation_func = MagicMock()
        
        # Mock the strategy methods
        self.optimizer._apply_basic_optimizations = MagicMock()
        self.optimizer._apply_aggressive_optimizations = MagicMock()
        
        # Set different optimization levels and test
        inputs = {"test": "data"}
        resource_allocation = ResourceAllocation(cpu_cores=1.0, memory_mb=512)
        
        # Test basic level
        self.optimizer.optimization_level = OptimizationLevel.BASIC
        self.optimizer.optimize_computation(computation_func, inputs, cacheable=False)
        self.optimizer._apply_basic_optimizations.assert_called_once()
        self.optimizer._apply_aggressive_optimizations.assert_not_called()
        
        # Reset mocks
        self.optimizer._apply_basic_optimizations.reset_mock()
        self.optimizer._apply_aggressive_optimizations.reset_mock()
        
        # Test aggressive level
        self.optimizer.optimization_level = OptimizationLevel.AGGRESSIVE
        self.optimizer.optimize_computation(computation_func, inputs, cacheable=False)
        self.optimizer._apply_basic_optimizations.assert_not_called()
        self.optimizer._apply_aggressive_optimizations.assert_called_once()


class TestOptimizeFunctionDecorator(unittest.TestCase):
    """Tests for the optimize_function decorator."""
    
    def test_decorator_functionality(self):
        """Test that the decorator works correctly."""
        # Create a mock performance optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.optimize_computation.return_value = {"result": 42}
        
        # Patch the get_performance_optimizer function
        with patch('orchestration.performance.performance_optimizer.get_performance_optimizer', 
                  return_value=mock_optimizer):
            
            # Define a function with the decorator
            @optimize_function(OptimizationLevel.BASIC)
            def example_func(inputs):
                return inputs.get("value", 0) * 2
            
            # Call the decorated function
            inputs = {"value": 10}
            result = example_func(inputs)
            
            # Verify the optimizer was used
            mock_optimizer.optimize_computation.assert_called_once_with(
                example_func.__wrapped__, inputs)
            self.assertEqual(result, {"result": 42})


if __name__ == '__main__':
    unittest.main()

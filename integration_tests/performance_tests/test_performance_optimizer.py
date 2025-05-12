"""
Integration tests for the Performance Optimization system.

These tests verify that the performance optimization components work together correctly,
including dynamic optimization level adjustment, bottleneck detection, and metrics visualization.
"""

import pytest
import time
import threading
import random
from unittest.mock import patch, MagicMock

from orchestration.performance.performance_optimizer import PerformanceOptimizer
from orchestration.performance.resource_manager import ResourceManager
from orchestration.monitoring.metrics_visualizer import MetricsVisualizer
from math_processing.computation.computation_cache import ComputationCache

class TestPerformanceOptimization:
    """Test suite for performance optimization components."""
    
    @pytest.fixture
    def mock_resource_manager(self):
        """Create a mock resource manager."""
        mock = MagicMock(spec=ResourceManager)
        
        # Configure the mock to return realistic system load values
        mock.get_system_load.return_value = {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_percent": 40.0
        }
        
        return mock
    
    @pytest.fixture
    def mock_computation_cache(self):
        """Create a mock computation cache."""
        mock = MagicMock(spec=ComputationCache)
        
        # Configure the mock to return realistic cache metrics
        mock.get_metrics.return_value = {
            "size": 100,
            "entries": 45,
            "hits": 120,
            "misses": 80,
            "hit_ratio": 0.6
        }
        
        return mock
    
    @pytest.fixture
    def performance_optimizer(self, mock_resource_manager, mock_computation_cache):
        """Create a performance optimizer with mocked dependencies."""
        with patch('orchestration.performance.performance_optimizer.ResourceManager', 
                  return_value=mock_resource_manager):
            with patch('orchestration.performance.performance_optimizer.ComputationCache', 
                      return_value=mock_computation_cache):
                # Create sample configuration
                config = {
                    "monitoring_interval": 1,  # Fast interval for testing
                    "high_cpu_threshold": 80.0,
                    "high_memory_threshold": 80.0,
                    "slow_response_threshold": 2.0,
                    "low_cache_hit_threshold": 0.4,
                    "slow_query_threshold": 0.5
                }
                
                optimizer = PerformanceOptimizer(config)
                
                # Add some sample metrics
                for i in range(100):
                    # CPU usage oscillating between 40% and 95%
                    cpu_usage = 40 + 55 * (0.5 + 0.5 * np.sin(i / 10))
                    # Memory usage gradually increasing
                    memory_usage = min(95, 40 + i * 0.5)
                    # Add metrics with timestamps
                    timestamp = time.time() - (100 - i) * 60  # Last 100 minutes
                    optimizer.metrics["cpu_usage"].append((timestamp, cpu_usage))
                    optimizer.metrics["memory_usage"].append((timestamp, memory_usage))
                
                # Add sample response times (between 0.1s and 3.0s)
                optimizer.metrics["response_times"] = [
                    random.uniform(0.1, 3.0) for _ in range(100)
                ]
                
                # Add sample optimization level history
                timestamp_base = time.time() - 86400  # 24 hours ago
                optimizer.metrics["optimization_level_history"] = [
                    (timestamp_base + 3600 * i, level) 
                    for i, level in enumerate([0, 1, 2, 1, 2, 1, 1, 0])
                ]
                
                yield optimizer
                
                # Cleanup
                optimizer.shutdown()
    
    @pytest.fixture
    def metrics_visualizer(self, performance_optimizer):
        """Create a metrics visualizer with the performance optimizer."""
        return MetricsVisualizer(performance_optimizer)
    
    def test_performance_optimizer_initialization(self, performance_optimizer):
        """Test that the performance optimizer initializes correctly."""
        assert performance_optimizer is not None
        assert performance_optimizer.current_optimization_level == performance_optimizer.OPTIMIZATION_LEVEL_BALANCED
        assert performance_optimizer.thresholds["high_cpu"] == 80.0
        assert performance_optimizer.thresholds["slow_response"] == 2.0
    
    def test_optimize_request_handling(self, performance_optimizer):
        """Test request handling optimization."""
        # Test with computation request
        computation_settings = performance_optimizer.optimize_request_handling(
            request_type="computation",
            payload_size=1000
        )
        assert computation_settings["parallel_processing"] is True
        
        # Test with visualization request
        visualization_settings = performance_optimizer.optimize_request_handling(
            request_type="visualization",
            payload_size=1000
        )
        assert visualization_settings["max_resources"]["memory_mb"] == 1536
        
        # Test with large payload
        large_payload_settings = performance_optimizer.optimize_request_handling(
            request_type="computation",
            payload_size=2 * 1024 * 1024  # 2MB
        )
        assert large_payload_settings["max_resources"]["memory_mb"] > computation_settings["max_resources"]["memory_mb"]
        assert large_payload_settings["timeout"] > computation_settings["timeout"]
    
    def test_optimize_computation(self, performance_optimizer, mock_resource_manager):
        """Test computation optimization."""
        # Configure mock to simulate high CPU load
        mock_resource_manager.get_system_load.return_value = {
            "cpu_percent": 85.0,
            "memory_percent": 50.0
        }
        
        # Test with complex calculus
        calculus_settings = performance_optimizer.optimize_computation(
            expression_type="calculus",
            complexity=8
        )
        assert "symbolic_first" in calculus_settings["optimization_techniques"]
        assert calculus_settings["parallel_processing"] is True
        # Should reduce parallelism due to high CPU load
        assert calculus_settings["max_processors"] <= 2
        
        # Configure mock to simulate high memory load
        mock_resource_manager.get_system_load.return_value = {
            "cpu_percent": 50.0,
            "memory_percent": 85.0
        }
        
        # Test with linear algebra
        linear_algebra_settings = performance_optimizer.optimize_computation(
            expression_type="linear_algebra",
            complexity=6
        )
        assert "matrix_optimization" in linear_algebra_settings["optimization_techniques"]
        assert "memory_efficient" in linear_algebra_settings["optimization_techniques"]
    
    def test_detect_bottlenecks(self, performance_optimizer, mock_resource_manager, mock_computation_cache):
        """Test bottleneck detection."""
        # Configure mocks to simulate multiple bottlenecks
        mock_resource_manager.get_system_load.return_value = {
            "cpu_percent": 90.0,  # High CPU
            "memory_percent": 85.0  # High memory
        }
        
        mock_computation_cache.get_metrics.return_value = {
            "size": 100,
            "entries": 45,
            "hits": 30,
            "misses": 70,
            "hit_ratio": 0.3  # Low cache hit ratio
        }
        
        # Add slow response times
        performance_optimizer.metrics["response_times"] = [3.5, 4.2, 3.8, 4.0, 3.7]
        
        # Detect bottlenecks
        bottlenecks = performance_optimizer.detect_bottlenecks()
        
        # Verify bottlenecks were detected
        assert len(bottlenecks) >= 3  # At least 3 bottlenecks
        bottleneck_types = [b["type"] for b in bottlenecks]
        assert "high_cpu_usage" in bottleneck_types
        assert "high_memory_usage" in bottleneck_types
        assert "low_cache_hit_ratio" in bottleneck_types
        assert "slow_response_time" in bottleneck_types
    
    def test_get_optimization_recommendations(self, performance_optimizer):
        """Test optimization recommendations generation."""
        # Force detection of bottlenecks
        with patch.object(performance_optimizer, 'detect_bottlenecks') as mock_detect:
            mock_detect.return_value = [
                {
                    "type": "high_cpu_usage",
                    "value": 90.0,
                    "threshold": 80.0,
                    "recommendation": "Consider scaling horizontally"
                },
                {
                    "type": "low_cache_hit_ratio",
                    "value": 0.3,
                    "threshold": 0.4,
                    "recommendation": "Adjust cache size"
                }
            ]
            
            recommendations = performance_optimizer.get_optimization_recommendations()
            
            # Verify recommendations
            assert len(recommendations) >= 2
            rec_issues = [r["issue"] for r in recommendations]
            assert "high_cpu_usage" in rec_issues
            assert "low_cache_hit_ratio" in rec_issues or "suboptimal_caching" in rec_issues
    
    def test_adjust_optimization_level(self, performance_optimizer, mock_resource_manager):
        """Test dynamic optimization level adjustment."""
        # Start with balanced level
        assert performance_optimizer.current_optimization_level == performance_optimizer.OPTIMIZATION_LEVEL_BALANCED
        
        # Simulate high load
        mock_resource_manager.get_system_load.return_value = {
            "cpu_percent": 95.0,
            "memory_percent": 90.0
        }
        performance_optimizer.metrics["response_times"] = [6.0, 5.5, 6.2, 5.8]
        
        # Adjust optimization level
        performance_optimizer.adjust_optimization_level()
        
        # Verify level was increased to aggressive
        assert performance_optimizer.current_optimization_level == performance_optimizer.OPTIMIZATION_LEVEL_AGGRESSIVE
        
        # Simulate low load
        mock_resource_manager.get_system_load.return_value = {
            "cpu_percent": 20.0,
            "memory_percent": 25.0
        }
        performance_optimizer.metrics["response_times"] = [0.2, 0.3, 0.25, 0.1]
        
        # Adjust optimization level
        performance_optimizer.adjust_optimization_level()
        
        # Verify level was decreased to minimal
        assert performance_optimizer.current_optimization_level == performance_optimizer.OPTIMIZATION_LEVEL_MINIMAL
    
    def test_metrics_visualizer_initialization(self, metrics_visualizer):
        """Test that the metrics visualizer initializes correctly."""
        assert metrics_visualizer is not None
        assert metrics_visualizer.chart_styles is not None
        assert "line_colors" in metrics_visualizer.chart_styles
    
    def test_create_system_load_chart(self, metrics_visualizer):
        """Test system load chart creation."""
        chart = metrics_visualizer.create_system_load_chart(time_period="1h")
        assert chart is not None
        assert isinstance(chart, str)
        assert chart.startswith("iVBOR") or chart.startswith("R0lGOD")  # Base64 image header
    
    def test_create_response_time_chart(self, metrics_visualizer):
        """Test response time chart creation."""
        chart = metrics_visualizer.create_response_time_chart(time_period="1h")
        assert chart is not None
        assert isinstance(chart, str)
        assert chart.startswith("iVBOR") or chart.startswith("R0lGOD")  # Base64 image header
    
    def test_create_optimization_level_chart(self, metrics_visualizer):
        """Test optimization level chart creation."""
        chart = metrics_visualizer.create_optimization_level_chart()
        assert chart is not None
        assert isinstance(chart, str)
        assert chart.startswith("iVBOR") or chart.startswith("R0lGOD")  # Base64 image header
    
    def test_create_performance_dashboard(self, metrics_visualizer):
        """Test creation of complete performance dashboard."""
        dashboard = metrics_visualizer.create_performance_dashboard()
        assert dashboard is not None
        assert isinstance(dashboard, dict)
        assert "system_load" in dashboard
        assert "response_time" in dashboard
        assert "optimization_level" in dashboard
        assert "bottleneck_analysis" in dashboard
        
        # Verify all charts are base64 images
        for chart_name, chart_data in dashboard.items():
            assert chart_data.startswith("iVBOR") or chart_data.startswith("R0lGOD")
    
    def test_create_grafana_dashboard_json(self, metrics_visualizer):
        """Test creation of Grafana dashboard configuration."""
        dashboard_json = metrics_visualizer.create_grafana_dashboard_json()
        assert dashboard_json is not None
        assert isinstance(dashboard_json, dict)
        assert "dashboard" in dashboard_json
        assert dashboard_json["dashboard"]["title"] == "Mathematical Multimodal LLM System Performance"
        assert len(dashboard_json["dashboard"]["panels"]) >= 5  # At least 5 panels

    @pytest.mark.integration
    def test_end_to_end_optimization(self, performance_optimizer, metrics_visualizer):
        """
        End-to-end test of the performance optimization system.
        
        This test simulates a complete cycle of performance optimization:
        1. Recording performance metrics
        2. Detecting bottlenecks
        3. Adjusting optimization level
        4. Generating visualization
        """
        # Record sample metrics
        for _ in range(10):
            performance_optimizer.record_metrics("response_time", random.uniform(0.5, 4.0))
        
        # Inject some system load metrics
        timestamp = time.time()
        performance_optimizer.metrics["cpu_usage"].append((timestamp, 85.0))
        performance_optimizer.metrics["memory_usage"].append((timestamp, 75.0))
        
        # Simulate database metrics
        query_optimizer = performance_optimizer.query_optimizer
        query_optimizer.get_metrics.return_value = {
            "average_query_time": 0.6,  # Slightly above threshold
            "slow_queries": ["findUsers", "aggregateStats"]
        }
        
        # Run a cycle of optimization
        bottlenecks = performance_optimizer.detect_bottlenecks()
        recommendations = performance_optimizer.get_optimization_recommendations()
        performance_optimizer.adjust_optimization_level()
        
        # Verify bottlenecks were detected
        assert len(bottlenecks) > 0
        
        # Verify recommendations were generated
        assert len(recommendations) > 0
        
        # Verify optimization level was adjusted appropriately
        assert performance_optimizer.current_optimization_level == performance_optimizer.OPTIMIZATION_LEVEL_AGGRESSIVE
        
        # Generate visualization
        dashboard = metrics_visualizer.create_performance_dashboard()
        
        # Verify dashboard was created
        assert len(dashboard) >= 4
        for chart_name, chart_data in dashboard.items():
            assert len(chart_data) > 1000  # Ensure charts have some content

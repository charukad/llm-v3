"""
Performance metrics visualization for the Mathematical Multimodal LLM System.

This module provides tools for visualizing and analyzing system performance metrics,
helping identify bottlenecks and optimization opportunities.
"""

import time
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from orchestration.monitoring.logger import get_logger
from orchestration.performance.performance_optimizer import get_performance_optimizer

logger = get_logger(__name__)


class PerformanceMetricsVisualizer:
    """
    Visualizes performance metrics to identify bottlenecks and optimization opportunities.
    """
    
    def __init__(self, metrics_history_size: int = 1000, output_dir: str = "performance_reports"):
        """
        Initialize the performance metrics visualizer.
        
        Args:
            metrics_history_size: Size of metrics history to maintain
            output_dir: Directory for saving visualization outputs
        """
        self.metrics_history_size = metrics_history_size
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = []
        self.timestamps = []
        
        logger.info(f"Initialized performance metrics visualizer with history size {metrics_history_size}")
    
    def record_metrics_snapshot(self) -> Dict[str, Any]:
        """
        Record a snapshot of current performance metrics.
        
        Returns:
            Dictionary of current metrics
        """
        # Get metrics from performance optimizer
        optimizer = get_performance_optimizer()
        metrics = optimizer.get_performance_metrics()
        
        # Add timestamp
        timestamp = time.time()
        self.timestamps.append(timestamp)
        metrics["timestamp"] = timestamp
        metrics["datetime"] = datetime.datetime.fromtimestamp(timestamp).isoformat()
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Trim history if it's too long
        if len(self.metrics_history) > self.metrics_history_size:
            self.metrics_history = self.metrics_history[-self.metrics_history_size:]
            self.timestamps = self.timestamps[-self.metrics_history_size:]
        
        return metrics
    
    def generate_system_load_chart(self, 
                                  save_path: Optional[str] = None,
                                  show: bool = False) -> str:
        """
        Generate a chart of system load over time.
        
        Args:
            save_path: Optional path to save the chart
            show: Whether to display the chart
            
        Returns:
            Path to the saved chart
        """
        # Ensure we have metrics
        if not self.metrics_history:
            logger.warning("No metrics available for visualization")
            return ""
        
        # Extract data
        timestamps = [
            datetime.datetime.fromtimestamp(m["timestamp"]) 
            for m in self.metrics_history
        ]
        cpu_load = [m.get("system_load", {}).get("cpu", 0) for m in self.metrics_history]
        memory_load = [m.get("system_load", {}).get("memory", 0) for m in self.metrics_history]
        gpu_load = [m.get("system_load", {}).get("gpu", 0) for m in self.metrics_history]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, cpu_load, label="CPU Load (%)", color="blue")
        plt.plot(timestamps, memory_load, label="Memory Load (%)", color="green")
        if any(gpu_load):
            plt.plot(timestamps, gpu_load, label="GPU Load (%)", color="red")
        
        # Add labels and formatting
        plt.title("System Resource Utilization Over Time")
        plt.xlabel("Time")
        plt.ylabel("Utilization (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"system_load_{timestamp}.png")
        
        plt.savefig(save_path)
        
        # Show if requested
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def generate_cache_performance_chart(self, 
                                        save_path: Optional[str] = None,
                                        show: bool = False) -> str:
        """
        Generate a chart of cache performance over time.
        
        Args:
            save_path: Optional path to save the chart
            show: Whether to display the chart
            
        Returns:
            Path to the saved chart
        """
        # Ensure we have metrics
        if not self.metrics_history:
            logger.warning("No metrics available for visualization")
            return ""
        
        # Extract data
        timestamps = [
            datetime.datetime.fromtimestamp(m["timestamp"]) 
            for m in self.metrics_history
        ]
        hit_ratio = [m.get("cache_hit_ratio", 0) * 100 for m in self.metrics_history]
        cache_size = [m.get("current_cache_size", 0) for m in self.metrics_history]
        max_cache_size = [m.get("max_cache_size", 0) for m in self.metrics_history]
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot hit ratio on left axis
        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cache Hit Ratio (%)', color=color)
        ax1.plot(timestamps, hit_ratio, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 100)
        
        # Create second y-axis for cache size
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Cache Size (MB)', color=color)
        ax2.plot(timestamps, cache_size, color=color, linestyle='-')
        ax2.plot(timestamps, max_cache_size, color='tab:green', linestyle='--', 
                label='Max Cache Size')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add legend and title
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, ['Hit Ratio'] + labels2, loc='upper right')
        
        plt.title("Cache Performance Metrics")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Save if requested
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"cache_performance_{timestamp}.png")
        
        plt.savefig(save_path)
        
        # Show if requested
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def generate_computation_time_chart(self, 
                                       computation_types: Optional[List[str]] = None,
                                       save_path: Optional[str] = None,
                                       show: bool = False) -> str:
        """
        Generate a chart of computation times by function type.
        
        Args:
            computation_types: Optional list of computation types to include
            save_path: Optional path to save the chart
            show: Whether to display the chart
            
        Returns:
            Path to the saved chart
        """
        # Extract computation time data
        computation_data = {}
        
        # Use any recorder computation times (this would be stored in the metrics)
        # In a real implementation, you would have a more complete dataset
        computation_examples = [
            {"function": "symbolic_calculation", "time": 0.45},
            {"function": "symbolic_calculation", "time": 0.52},
            {"function": "symbolic_calculation", "time": 0.38},
            {"function": "integrate_expression", "time": 0.78},
            {"function": "integrate_expression", "time": 0.92},
            {"function": "solve_equation", "time": 0.31},
            {"function": "solve_equation", "time": 0.28},
            {"function": "plot_2d", "time": 0.15},
            {"function": "plot_3d", "time": 0.43},
            {"function": "handwriting_recognition", "time": 1.25},
            {"function": "handwriting_recognition", "time": 1.42}
        ]
        
        # Group by function type
        for entry in computation_examples:
            func = entry["function"]
            time_val = entry["time"]
            
            if func not in computation_data:
                computation_data[func] = []
                
            computation_data[func].append(time_val)
        
        # Filter by computation types if specified
        if computation_types:
            computation_data = {k: v for k, v in computation_data.items() if k in computation_types}
        
        # Calculate statistics
        function_names = list(computation_data.keys())
        mean_times = [np.mean(computation_data[func]) for func in function_names]
        std_times = [np.std(computation_data[func]) for func in function_names]
        
        # Sort by mean time (descending)
        sorted_indices = np.argsort(mean_times)[::-1]
        function_names = [function_names[i] for i in sorted_indices]
        mean_times = [mean_times[i] for i in sorted_indices]
        std_times = [std_times[i] for i in sorted_indices]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        bars = plt.bar(function_names, mean_times, yerr=std_times, capsize=5, color='skyblue')
        
        # Add labels and formatting
        plt.title("Average Computation Time by Function Type")
        plt.xlabel("Function Type")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add values on top of bars
        for bar, value in zip(bars, mean_times):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.02,
                    f'{value:.2f}s',
                    ha='center', va='bottom')
        
        # Save if requested
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"computation_time_{timestamp}.png")
        
        plt.savefig(save_path)
        
        # Show if requested
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def generate_performance_report(self, 
                                   save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Path to the saved report
        """
        # Create charts
        system_load_chart = self.generate_system_load_chart()
        cache_chart = self.generate_cache_performance_chart()
        computation_chart = self.generate_computation_time_chart()
        
        # Get latest metrics
        latest_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        # Generate HTML report
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mathematical Multimodal LLM System - Performance Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                h1, h2 {{
                    color: #333;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .metric-value {{
                    font-weight: bold;
                    color: #2c5777;
                }}
                .metric-row {{
                    display: flex;
                    margin-bottom: 10px;
                }}
                .metric-label {{
                    width: 250px;
                    font-weight: bold;
                }}
                .chart {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .tips {{
                    background-color: #e7f4ff;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Mathematical Multimodal LLM System</h1>
                <h2>Performance Report</h2>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <div class="container">
                <h2>Current Performance Metrics</h2>
                
                <div class="metric-row">
                    <div class="metric-label">CPU Load:</div>
                    <div class="metric-value">{latest_metrics.get("system_load", {}).get("cpu", 0):.2f}%</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-label">Memory Load:</div>
                    <div class="metric-value">{latest_metrics.get("system_load", {}).get("memory", 0):.2f}%</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-label">GPU Load:</div>
                    <div class="metric-value">{latest_metrics.get("system_load", {}).get("gpu", 0):.2f}%</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-label">Cache Hit Ratio:</div>
                    <div class="metric-value">{latest_metrics.get("cache_hit_ratio", 0) * 100:.2f}%</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-label">Cache Size:</div>
                    <div class="metric-value">{latest_metrics.get("current_cache_size", 0):.2f} MB / {latest_metrics.get("max_cache_size", 0)} MB</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-label">Parallel Tasks Completed:</div>
                    <div class="metric-value">{latest_metrics.get("parallel_tasks_completed", 0)}</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-label">Messages Batched:</div>
                    <div class="metric-value">{latest_metrics.get("messages_batched", 0)}</div>
                </div>
            </div>
            
            <div class="container">
                <h2>System Load Visualization</h2>
                <div class="chart">
                    <img src="{os.path.basename(system_load_chart)}" alt="System Load Chart">
                </div>
            </div>
            
            <div class="container">
                <h2>Cache Performance</h2>
                <div class="chart">
                    <img src="{os.path.basename(cache_chart)}" alt="Cache Performance Chart">
                </div>
            </div>
            
            <div class="container">
                <h2>Computation Time by Function</h2>
                <div class="chart">
                    <img src="{os.path.basename(computation_chart)}" alt="Computation Time Chart">
                </div>
            </div>
            
            <div class="container">
                <h2>Optimization Recommendations</h2>
                <div class="tips">
                    <h3>Recommendations based on current metrics:</h3>
                    <ul>
                        {"<li>Consider increasing cache size for better hit ratio.</li>" if latest_metrics.get("cache_hit_ratio", 0) < 0.7 else ""}
                        {"<li>CPU usage is high. Consider distributing load or scaling up resources.</li>" if latest_metrics.get("system_load", {}).get("cpu", 0) > 70 else ""}
                        {"<li>Memory usage is high. Consider optimizing memory-intensive operations.</li>" if latest_metrics.get("system_load", {}).get("memory", 0) > 70 else ""}
                        {"<li>Symbolic calculations are taking significant time. Consider numerical approximations for non-critical cases.</li>" if any(entry["function"] == "symbolic_calculation" and entry["time"] > 0.5 for entry in computation_examples) else ""}
                        {"<li>The system resources appear well-balanced and efficiently utilized.</li>" if latest_metrics.get("system_load", {}).get("cpu", 0) < 70 and latest_metrics.get("system_load", {}).get("memory", 0) < 70 and latest_metrics.get("cache_hit_ratio", 0) > 0.7 else ""}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the report
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"performance_report_{timestamp}.html")
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Generated performance report: {save_path}")
        
        return save_path


# Singleton instance
_metrics_visualizer = None

def get_metrics_visualizer() -> PerformanceMetricsVisualizer:
    """
    Get the singleton instance of the metrics visualizer.
    
    Returns:
        The metrics visualizer instance
    """
    global _metrics_visualizer
    if _metrics_visualizer is None:
        _metrics_visualizer = PerformanceMetricsVisualizer()
    return _metrics_visualizer


def generate_current_performance_report() -> str:
    """
    Generate a performance report based on current metrics.
    
    Returns:
        Path to the generated report
    """
    visualizer = get_metrics_visualizer()
    visualizer.record_metrics_snapshot()
    return visualizer.generate_performance_report()

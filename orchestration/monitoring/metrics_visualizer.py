"""
Metrics Visualizer

This module provides visualization functionality for performance metrics,
enabling better understanding of system performance and bottlenecks.
"""

import logging
import time
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from orchestration.performance.performance_optimizer import PerformanceOptimizer

# Configure logger
logger = logging.getLogger(__name__)

class MetricsVisualizer:
    """
    Visualization system for performance metrics with various chart types
    and integration with monitoring dashboards.
    """
    
    def __init__(self, performance_optimizer: PerformanceOptimizer):
        """
        Initialize the metrics visualizer with a reference to the performance optimizer.
        
        Args:
            performance_optimizer: The performance optimizer instance
        """
        self.performance_optimizer = performance_optimizer
        self.chart_styles = {
            "background_color": "#f5f5f5",
            "grid_color": "#dddddd",
            "text_color": "#333333",
            "line_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            "bar_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            "title_font_size": 14,
            "axis_font_size": 12,
            "legend_font_size": 10,
        }
        logger.info("Metrics visualizer initialized")
    
    def create_system_load_chart(self, time_period: str = "1h") -> str:
        """
        Create a chart showing CPU and memory usage over time.
        
        Args:
            time_period: Time period to display (e.g., '1h', '1d', '1w')
            
        Returns:
            Base64-encoded PNG image of the chart
        """
        # Get time range based on period
        end_time = time.time()
        if time_period == "1h":
            start_time = end_time - 3600  # 1 hour
        elif time_period == "1d":
            start_time = end_time - 86400  # 1 day
        elif time_period == "1w":
            start_time = end_time - 604800  # 1 week
        else:
            start_time = end_time - 3600  # Default to 1 hour
        
        # Filter metrics for the selected time period
        cpu_data = [(t, v) for t, v in self.performance_optimizer.metrics["cpu_usage"] if t >= start_time and t <= end_time]
        memory_data = [(t, v) for t, v in self.performance_optimizer.metrics["memory_usage"] if t >= start_time and t <= end_time]
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        plt.rcParams['axes.facecolor'] = self.chart_styles["background_color"]
        plt.grid(color=self.chart_styles["grid_color"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # CPU usage line
        if cpu_data:
            cpu_times = [datetime.datetime.fromtimestamp(t) for t, _ in cpu_data]
            cpu_values = [v for _, v in cpu_data]
            plt.plot(
                cpu_times, 
                cpu_values, 
                color=self.chart_styles["line_colors"][0], 
                linewidth=2, 
                marker='o', 
                markersize=3, 
                label="CPU Usage (%)"
            )
        
        # Memory usage line
        if memory_data:
            memory_times = [datetime.datetime.fromtimestamp(t) for t, _ in memory_data]
            memory_values = [v for _, v in memory_data]
            plt.plot(
                memory_times, 
                memory_values, 
                color=self.chart_styles["line_colors"][1], 
                linewidth=2, 
                marker='s', 
                markersize=3, 
                label="Memory Usage (%)"
            )
        
        # Configure date formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M' if time_period == "1h" else '%m-%d %H:%M'))
        if time_period == "1d":
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
        elif time_period == "1w":
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        else:
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        
        plt.gcf().autofmt_xdate()
        
        # Add labels and title
        plt.xlabel('Time', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.ylabel('Usage (%)', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.title(f'System Resource Usage - Last {time_period}', 
                  fontsize=self.chart_styles["title_font_size"], 
                  color=self.chart_styles["text_color"])
        
        # Add thresholds
        plt.axhline(y=80, color='#ff0000', linestyle='--', alpha=0.7, label="High Load Threshold")
        
        # Add legend
        plt.legend(loc='upper left', fontsize=self.chart_styles["legend_font_size"])
        
        # Set y-axis range
        plt.ylim(0, 105)
        
        # Convert plot to base64 PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def create_response_time_chart(self, time_period: str = "1h") -> str:
        """
        Create a chart showing response times over time.
        
        Args:
            time_period: Time period to display (e.g., '1h', '1d', '1w')
            
        Returns:
            Base64-encoded PNG image of the chart
        """
        # We only have response time values without timestamps in the current metrics structure
        # For visualization purposes, we'll create estimated timestamps based on the number of responses
        response_times = self.performance_optimizer.metrics["response_times"]
        
        if not response_times:
            # Create an empty chart if no data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No response time data available", 
                     fontsize=14, ha='center', va='center', color=self.chart_styles["text_color"])
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create estimated timestamps
        end_time = time.time()
        
        # Determine start time based on time period
        if time_period == "1h":
            start_time = end_time - 3600  # 1 hour
        elif time_period == "1d":
            start_time = end_time - 86400  # 1 day
        elif time_period == "1w":
            start_time = end_time - 604800  # 1 week
        else:
            start_time = end_time - 3600  # Default to 1 hour
        
        # Create evenly spaced timestamps between start_time and end_time
        timestamps = np.linspace(start_time, end_time, min(len(response_times), 100))
        
        # Select a subset of response times if there are too many
        if len(response_times) > 100:
            indices = np.linspace(0, len(response_times) - 1, 100, dtype=int)
            selected_response_times = [response_times[i] for i in indices]
        else:
            selected_response_times = response_times
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        plt.rcParams['axes.facecolor'] = self.chart_styles["background_color"]
        plt.grid(color=self.chart_styles["grid_color"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Plot response times
        plt.plot(
            [datetime.datetime.fromtimestamp(t) for t in timestamps], 
            selected_response_times, 
            color=self.chart_styles["line_colors"][2], 
            linewidth=2, 
            marker='o', 
            markersize=3,
            label="Response Time"
        )
        
        # Add a rolling average line
        window_size = min(10, len(selected_response_times))
        rolling_avg = np.convolve(selected_response_times, np.ones(window_size)/window_size, mode='valid')
        plt.plot(
            [datetime.datetime.fromtimestamp(t) for t in timestamps[window_size-1:]], 
            rolling_avg, 
            color=self.chart_styles["line_colors"][3], 
            linewidth=2, 
            label=f"{window_size}-point Moving Average"
        )
        
        # Configure date formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M' if time_period == "1h" else '%m-%d %H:%M'))
        if time_period == "1d":
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
        elif time_period == "1w":
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        else:
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        
        plt.gcf().autofmt_xdate()
        
        # Add labels and title
        plt.xlabel('Time', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.ylabel('Response Time (seconds)', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.title(f'Response Time Trend - Last {time_period}', 
                  fontsize=self.chart_styles["title_font_size"], 
                  color=self.chart_styles["text_color"])
        
        # Add threshold for slow responses
        plt.axhline(y=self.performance_optimizer.thresholds["slow_response"], 
                   color='#ff0000', linestyle='--', alpha=0.7, 
                   label=f"Slow Response Threshold ({self.performance_optimizer.thresholds['slow_response']}s)")
        
        # Add legend
        plt.legend(loc='upper left', fontsize=self.chart_styles["legend_font_size"])
        
        # Set y-axis range with a small buffer above max value
        max_value = max(selected_response_times) * 1.2  # 20% buffer
        plt.ylim(0, max_value)
        
        # Convert plot to base64 PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def create_optimization_level_chart(self) -> str:
        """
        Create a chart showing the optimization level changes over time.
        
        Returns:
            Base64-encoded PNG image of the chart
        """
        optimization_history = self.performance_optimizer.metrics["optimization_level_history"]
        
        if not optimization_history:
            # Create an empty chart if no data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No optimization level data available", 
                     fontsize=14, ha='center', va='center', color=self.chart_styles["text_color"])
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Extract data
        timestamps = [datetime.datetime.fromtimestamp(t) for t, _ in optimization_history]
        levels = [lvl for _, lvl in optimization_history]
        
        # Map numeric levels to readable names
        level_names = {
            0: "Minimal",
            1: "Balanced",
            2: "Aggressive"
        }
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        plt.rcParams['axes.facecolor'] = self.chart_styles["background_color"]
        plt.grid(color=self.chart_styles["grid_color"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Plot optimization levels as a step chart
        plt.step(
            timestamps, 
            levels, 
            where='post',
            color=self.chart_styles["line_colors"][4], 
            linewidth=2, 
            marker='o', 
            markersize=5
        )
        
        # Configure date formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.gcf().autofmt_xdate()
        
        # Configure y-axis to show level names
        plt.yticks([0, 1, 2], [level_names[0], level_names[1], level_names[2]])
        
        # Add labels and title
        plt.xlabel('Time', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.ylabel('Optimization Level', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.title('Optimization Level Changes', 
                  fontsize=self.chart_styles["title_font_size"], 
                  color=self.chart_styles["text_color"])
        
        # Set y-axis range with a small buffer
        plt.ylim(-0.1, 2.1)
        
        # Convert plot to base64 PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def create_bottleneck_analysis_chart(self) -> str:
        """
        Create a chart showing detected bottlenecks by type.
        
        Returns:
            Base64-encoded PNG image of the chart
        """
        # Get current bottlenecks
        bottlenecks = self.performance_optimizer.detect_bottlenecks()
        
        if not bottlenecks:
            # Create an empty chart if no bottlenecks
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No bottlenecks detected", 
                     fontsize=14, ha='center', va='center', color=self.chart_styles["text_color"])
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Count bottlenecks by type
        bottleneck_types = {}
        for bottleneck in bottlenecks:
            bottleneck_type = bottleneck["type"]
            if bottleneck_type in bottleneck_types:
                bottleneck_types[bottleneck_type] = bottleneck_types[bottleneck_type] + 1
            else:
                bottleneck_types[bottleneck_type] = 1
        
        # Clean up names for display
        display_names = {
            "high_cpu_usage": "High CPU Usage",
            "high_memory_usage": "High Memory Usage",
            "slow_response_time": "Slow Response Time",
            "low_cache_hit_ratio": "Low Cache Hit Ratio",
            "slow_database_queries": "Slow Database Queries"
        }
        
        # Prepare data for chart
        labels = [display_names.get(btype, btype) for btype in bottleneck_types.keys()]
        values = list(bottleneck_types.values())
        
        # Calculate severity values by comparing current values to thresholds
        severity = []
        for bottleneck in bottlenecks:
            if "value" in bottleneck and "threshold" in bottleneck:
                if bottleneck["type"] == "low_cache_hit_ratio":
                    # Inverse ratio for low cache hit ratio
                    severity.append(bottleneck["threshold"] / max(0.01, bottleneck["value"]))
                else:
                    severity.append(bottleneck["value"] / bottleneck["threshold"])
        
        # Average severity by bottleneck type
        severity_by_type = {}
        for bottleneck in bottlenecks:
            bottleneck_type = bottleneck["type"]
            if "value" in bottleneck and "threshold" in bottleneck:
                if bottleneck_type == "low_cache_hit_ratio":
                    # Inverse ratio for low cache hit ratio
                    sev = bottleneck["threshold"] / max(0.01, bottleneck["value"])
                else:
                    sev = bottleneck["value"] / bottleneck["threshold"]
                
                if bottleneck_type in severity_by_type:
                    severity_by_type[bottleneck_type].append(sev)
                else:
                    severity_by_type[bottleneck_type] = [sev]
        
        # Calculate average severity for each type
        avg_severity = []
        for btype in bottleneck_types.keys():
            if btype in severity_by_type:
                avg_severity.append(sum(severity_by_type[btype]) / len(severity_by_type[btype]))
            else:
                avg_severity.append(1.0)  # Default severity
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        plt.rcParams['axes.facecolor'] = self.chart_styles["background_color"]
        
        # Create a horizontal bar chart
        bars = plt.barh(
            labels, 
            values, 
            color=[plt.cm.RdYlGn_r(min(1.0, sev/2)) for sev in avg_severity],  # Color by severity
            alpha=0.7
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.1, 
                bar.get_y() + bar.get_height()/2, 
                f"{width}", 
                va='center', 
                fontsize=10
            )
        
        # Add labels and title
        plt.xlabel('Count', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.ylabel('Bottleneck Type', fontsize=self.chart_styles["axis_font_size"], color=self.chart_styles["text_color"])
        plt.title('Performance Bottlenecks Analysis', 
                  fontsize=self.chart_styles["title_font_size"], 
                  color=self.chart_styles["text_color"])
        
        # Add colorbar to indicate severity
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(1, 2))
        sm.set_array([])
        cbar = plt.colorbar(sm, orientation='vertical', pad=0.01)
        cbar.set_label('Severity (Ratio to Threshold)', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to base64 PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def create_performance_dashboard(self) -> Dict[str, str]:
        """
        Create a complete performance dashboard with multiple charts.
        
        Returns:
            Dictionary of chart names to base64-encoded PNG images
        """
        logger.info("Generating performance dashboard")
        dashboard = {}
        
        try:
            # System load chart
            dashboard["system_load"] = self.create_system_load_chart("1d")
            
            # Response time chart
            dashboard["response_time"] = self.create_response_time_chart("1d")
            
            # Optimization level chart
            dashboard["optimization_level"] = self.create_optimization_level_chart()
            
            # Bottleneck analysis chart
            dashboard["bottleneck_analysis"] = self.create_bottleneck_analysis_chart()
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            # Create an error chart
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error generating charts: {str(e)}", 
                     fontsize=14, ha='center', va='center', color='red')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            buffer.seek(0)
            error_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Add error image to any missing charts
            for chart_type in ["system_load", "response_time", "optimization_level", "bottleneck_analysis"]:
                if chart_type not in dashboard:
                    dashboard[chart_type] = error_image
        
        return dashboard
    
    def create_grafana_dashboard_json(self) -> Dict[str, Any]:
        """
        Generate a Grafana dashboard configuration based on the metrics.
        
        Returns:
            Grafana dashboard configuration JSON
        """
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Mathematical Multimodal LLM System Performance",
                "tags": ["performance", "llm", "mathematical"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 1,
                "refresh": "5s",
                "panels": []
            },
            "overwrite": True
        }
        
        # System load panel
        dashboard["dashboard"]["panels"].append({
            "title": "System Resource Usage",
            "type": "graph",
            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
            "id": 1,
            "targets": [
                {"expr": "system_cpu_usage", "legendFormat": "CPU Usage (%)"},
                {"expr": "system_memory_usage", "legendFormat": "Memory Usage (%)"}
            ],
            "yaxes": [
                {"format": "percent", "min": 0, "max": 100},
                {"format": "short", "show": False}
            ],
            "thresholds": [
                {"value": 80, "colorMode": "critical", "op": "gt", "line": True, "fill": False}
            ]
        })
        
        # Response time panel
        dashboard["dashboard"]["panels"].append({
            "title": "Response Times",
            "type": "graph",
            "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
            "id": 2,
            "targets": [
                {"expr": "request_response_time", "legendFormat": "Response Time"},
                {"expr": "request_response_time_avg", "legendFormat": "Average Response Time"}
            ],
            "yaxes": [
                {"format": "s", "min": 0},
                {"format": "short", "show": False}
            ],
            "thresholds": [
                {"value": self.performance_optimizer.thresholds["slow_response"], 
                 "colorMode": "critical", "op": "gt", "line": True, "fill": False}
            ]
        })
        
        # Cache hit ratio panel
        dashboard["dashboard"]["panels"].append({
            "title": "Cache Hit Ratio",
            "type": "gauge",
            "gridPos": {"x": 0, "y": 8, "w": 6, "h": 6},
            "id": 3,
            "targets": [
                {"expr": "cache_hit_ratio", "legendFormat": ""}
            ],
            "options": {
                "thresholds": [
                    {"color": "red", "value": 0},
                    {"color": "yellow", "value": 0.4},
                    {"color": "green", "value": 0.7}
                ],
                "max": 1,
                "min": 0,
                "unit": "percentunit"
            }
        })
        
        # Optimization level panel
        dashboard["dashboard"]["panels"].append({
            "title": "Current Optimization Level",
            "type": "stat",
            "gridPos": {"x": 6, "y": 8, "w": 6, "h": 6},
            "id": 4,
            "targets": [
                {"expr": "optimization_level", "legendFormat": ""}
            ],
            "options": {
                "colorMode": "value",
                "mappings": [
                    {"type": "value", "id": 1, "text": "Minimal", "value": "0"},
                    {"type": "value", "id": 2, "text": "Balanced", "value": "1"},
                    {"type": "value", "id": 3, "text": "Aggressive", "value": "2"}
                ],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "blue", "value": 0},
                        {"color": "green", "value": 1},
                        {"color": "red", "value": 2}
                    ]
                }
            }
        })
        
        # Database query time panel
        dashboard["dashboard"]["panels"].append({
            "title": "Database Query Times",
            "type": "graph",
            "gridPos": {"x": 12, "y": 8, "w": 12, "h": 6},
            "id": 5,
            "targets": [
                {"expr": "database_query_time_avg", "legendFormat": "Average Query Time"},
                {"expr": "database_query_time_p95", "legendFormat": "95th Percentile"}
            ],
            "yaxes": [
                {"format": "s", "min": 0},
                {"format": "short", "show": False}
            ],
            "thresholds": [
                {"value": self.performance_optimizer.thresholds["slow_query"], 
                 "colorMode": "critical", "op": "gt", "line": True, "fill": False}
            ]
        })
        
        return dashboard

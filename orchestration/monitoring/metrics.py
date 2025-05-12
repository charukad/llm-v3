"""
Metrics collection for the Mathematical Multimodal LLM System.

This module provides utilities for collecting and reporting various metrics
about the system's performance and behavior.
"""
import time
from typing import Dict, Any, List, Optional
import asyncio
import logging
import json
from collections import defaultdict
from dataclasses import dataclass, field
import threading
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricCounter:
    """A counter metric with increment/decrement functionality."""
    name: str
    value: int = 0
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    
    def increment(self, amount: int = 1):
        """Increment the counter."""
        self.value += amount
        
    def decrement(self, amount: int = 1):
        """Decrement the counter."""
        self.value -= amount
        
    def reset(self):
        """Reset the counter to zero."""
        self.value = 0


@dataclass
class MetricGauge:
    """A gauge metric that can go up or down."""
    name: str
    value: float = 0.0
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    
    def set(self, value: float):
        """Set the gauge to a specific value."""
        self.value = value
        
    def increment(self, amount: float = 1.0):
        """Increment the gauge."""
        self.value += amount
        
    def decrement(self, amount: float = 1.0):
        """Decrement the gauge."""
        self.value -= amount


@dataclass
class MetricHistogram:
    """A histogram metric for tracking distributions."""
    name: str
    description: str = ""
    buckets: List[float] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    sum: float = 0.0
    count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize bucket counts if not provided."""
        if not self.buckets:
            # Default buckets for latency in milliseconds
            self.buckets = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)
            
    def observe(self, value: float):
        """Record an observation."""
        self.sum += value
        self.count += 1
        
        # Find the appropriate bucket
        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self.counts[i] += 1
                return
                
        # If we get here, it belongs in the last bucket
        self.counts[-1] += 1
        
    def reset(self):
        """Reset the histogram."""
        self.counts = [0] * (len(self.buckets) + 1)
        self.sum = 0.0
        self.count = 0


class MetricsRegistry:
    """Registry for all metrics in the system."""
    def __init__(self):
        self.counters: Dict[str, MetricCounter] = {}
        self.gauges: Dict[str, MetricGauge] = {}
        self.histograms: Dict[str, MetricHistogram] = {}
        self._lock = threading.Lock()
        
    def counter(self, name: str, description: str = "", labels: Dict[str, str] = None) -> MetricCounter:
        """Get or create a counter metric."""
        key = self._get_key(name, labels)
        with self._lock:
            if key not in self.counters:
                self.counters[key] = MetricCounter(name, 0, description, labels or {})
            return self.counters[key]
            
    def gauge(self, name: str, description: str = "", labels: Dict[str, str] = None) -> MetricGauge:
        """Get or create a gauge metric."""
        key = self._get_key(name, labels)
        with self._lock:
            if key not in self.gauges:
                self.gauges[key] = MetricGauge(name, 0.0, description, labels or {})
            return self.gauges[key]
            
    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: List[float] = None,
        labels: Dict[str, str] = None
    ) -> MetricHistogram:
        """Get or create a histogram metric."""
        key = self._get_key(name, labels)
        with self._lock:
            if key not in self.histograms:
                self.histograms[key] = MetricHistogram(name, description, buckets or [], [], 0.0, 0, labels or {})
            return self.histograms[key]
            
    def _get_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Get a unique key for a metric based on name and labels."""
        if not labels:
            return name
            
        # Sort labels by key to ensure consistent keys
        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{name}[{label_str}]"
        
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics as a dictionary."""
        result = {
            "counters": [self._format_metric(c) for c in self.counters.values()],
            "gauges": [self._format_metric(g) for g in self.gauges.values()],
            "histograms": [self._format_histogram(h) for h in self.histograms.values()]
        }
        return result
        
    def _format_metric(self, metric) -> Dict[str, Any]:
        """Format a counter or gauge metric for output."""
        return {
            "name": metric.name,
            "value": metric.value,
            "description": metric.description,
            "labels": metric.labels
        }
        
    def _format_histogram(self, histogram) -> Dict[str, Any]:
        """Format a histogram metric for output."""
        buckets = []
        for i, bucket in enumerate(histogram.buckets):
            buckets.append({
                "le": bucket,
                "count": histogram.counts[i]
            })
        buckets.append({
            "le": "Inf",
            "count": histogram.counts[-1]
        })
        
        return {
            "name": histogram.name,
            "description": histogram.description,
            "buckets": buckets,
            "sum": histogram.sum,
            "count": histogram.count,
            "labels": histogram.labels
        }
        
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in the specified format."""
        metrics = self.get_all_metrics()
        if format.lower() == "json":
            return json.dumps(metrics, indent=2)
        else:
            return json.dumps(metrics, indent=2)  # Default to JSON for now


# Create a global registry
_registry = MetricsRegistry()

def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _registry


# Predefined metrics for the message bus
def setup_message_bus_metrics():
    """Set up default metrics for the message bus."""
    registry = get_registry()
    
    # Message counters
    registry.counter("message_bus.messages.total", "Total number of messages processed")
    registry.counter("message_bus.messages.errors", "Number of message processing errors")
    
    # Queue gauges
    registry.gauge("message_bus.queue.size", "Current size of the message queue")
    registry.gauge("message_bus.active_agents", "Number of active agents")
    
    # Latency histograms
    registry.histogram("message_bus.latency.processing", "Message processing latency (ms)")
    registry.histogram("message_bus.latency.routing", "Message routing latency (ms)")


# Helper functions for recording common metrics
def record_message_metrics(
    message_type: str,
    sender: str,
    recipient: str,
    size: int
):
    """Record metrics for a message."""
    registry = get_registry()
    
    # Increment total messages
    registry.counter("message_bus.messages.total").increment()
    
    # Increment message type counter
    registry.counter(
        "message_bus.messages.by_type",
        labels={"type": message_type}
    ).increment()
    
    # Track message size
    registry.histogram(
        "message_bus.message_size",
        buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
    ).observe(size)


def record_processing_time(duration_ms: float):
    """Record processing time for a message."""
    registry = get_registry()
    registry.histogram("message_bus.latency.processing").observe(duration_ms)


def record_error(error_type: str):
    """Record an error."""
    registry = get_registry()
    registry.counter("message_bus.messages.errors").increment()
    registry.counter(
        "message_bus.errors.by_type",
        labels={"type": error_type}
    ).increment()


# Initialize default metrics
setup_message_bus_metrics()

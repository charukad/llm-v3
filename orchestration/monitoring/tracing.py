"""
Distributed tracing implementation for the Multi-agent Communication Protocol (MCP).

This module provides tracing capabilities to track message flow through the system
and help with debugging, performance analysis, and monitoring.
"""
import time
import datetime
import uuid
from typing import Dict, Any, Optional, List
import json
from pydantic import BaseModel
import asyncio
import logging
from .logger import get_logger

logger = get_logger(__name__)


class SpanContext(BaseModel):
    """Context information for a trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = {}


class Span:
    """
    A span represents a single operation within a trace.
    
    Spans can be nested to represent a causal relationship between operations.
    """
    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.start_time = time.time()
        self.end_time = None
        self.metadata = metadata or {}
        self.children: List[Span] = []
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the span."""
        self.metadata[key] = value
        
    def end(self):
        """End the span."""
        self.end_time = time.time()
        
    def duration(self) -> Optional[float]:
        """Get the duration of the span in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000  # Convert to milliseconds
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the span to a dictionary."""
        result = {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration(),
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children]
        }
        return result
        
    def to_json(self) -> str:
        """Convert the span to JSON."""
        return json.dumps(self.to_dict(), default=str)
        
    def __enter__(self):
        """Enter the context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.end()
        if exc_type is not None:
            self.add_metadata("error", {
                "type": exc_type.__name__,
                "message": str(exc_val)
            })
            
    @staticmethod
    async def async_span(name: str, trace_id: Optional[str] = None, parent_span_id: Optional[str] = None, metadata: Dict[str, Any] = None):
        """Create a span for use with 'async with'."""
        span = Span(name, trace_id, parent_span_id, metadata)
        try:
            yield span
        except Exception as e:
            span.add_metadata("error", {
                "type": type(e).__name__,
                "message": str(e)
            })
            raise
        finally:
            span.end()


class Tracer:
    """
    Tracer for tracking the execution of operations across multiple agents.
    """
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans: Dict[str, Span] = {}
        self.trace_exporters: List[callable] = []
        
    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Span:
        """Start a new span."""
        span = Span(name, trace_id, parent_span_id, metadata)
        self.active_spans[span.span_id] = span
        logger.debug(f"Started span {span.span_id} for {name}")
        return span
        
    def end_span(self, span_id: str):
        """End a span by ID."""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end()
            del self.active_spans[span_id]
            
            # Export the completed span
            for exporter in self.trace_exporters:
                try:
                    exporter(span)
                except Exception as e:
                    logger.error(f"Error exporting span: {str(e)}")
                    
            logger.debug(f"Ended span {span_id}")
            return span
        else:
            logger.warning(f"Attempted to end unknown span: {span_id}")
            return None
            
    def add_exporter(self, exporter: callable):
        """Add a trace exporter function."""
        self.trace_exporters.append(exporter)
        
    def clear_exporters(self):
        """Clear all trace exporters."""
        self.trace_exporters = []
        
    def span(self, name: str, trace_id: Optional[str] = None, parent_span_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> Span:
        """Create a span context manager."""
        return self.start_span(name, trace_id, parent_span_id, metadata)


# Initialize a global tracer for the message bus
_global_tracer = Tracer("math_llm_system")

def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    return _global_tracer


# Example trace exporters
def console_exporter(span: Span):
    """Export a span to the console (for debugging)."""
    logger.debug(f"TRACE: {span.to_json()}")

def log_exporter(span: Span):
    """Export a span to the logs."""
    if span.duration() and span.duration() > 100:  # Only log spans longer than 100ms
        logger.info(f"TRACE[{span.trace_id}]: {span.name} - {span.duration():.2f}ms")

# Add default exporters
get_tracer().add_exporter(log_exporter)

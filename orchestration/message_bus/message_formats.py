"""
Comprehensive message format specification for the Multi-agent Communication Protocol (MCP).

This module defines the standard message formats used for communication between agents
in the Mathematical Multimodal LLM System.
"""
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import datetime
import uuid


class MessagePriority(str, Enum):
    """Priority levels for messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MessageType(str, Enum):
    """Types of messages in the system."""
    # Request types
    QUERY = "query"                           # General query
    COMPUTATION_REQUEST = "computation_request"  # Mathematical computation
    VISUALIZATION_REQUEST = "visualization_request"  # Visualization generation
    OCR_REQUEST = "ocr_request"              # OCR processing
    SEARCH_REQUEST = "search_request"        # External search
    
    # Response types
    QUERY_RESPONSE = "query_response"        # Response to general query
    COMPUTATION_RESULT = "computation_result"  # Result of computation
    VISUALIZATION_RESULT = "visualization_result"  # Generated visualization
    OCR_RESULT = "ocr_result"                # OCR processing result
    SEARCH_RESULT = "search_result"          # Search results
    ERROR = "error"                          # Error response
    
    # System messages
    HEARTBEAT = "heartbeat"                  # Agent heartbeat
    CAPABILITY_ADVERTISEMENT = "capability_advertisement"  # Agent capabilities
    STATUS_UPDATE = "status_update"          # Status updates
    LOG = "log"                              # Log messages
    METRICS = "metrics"                      # Performance metrics


class Route(BaseModel):
    """Routing information for a message."""
    sender: str
    recipient: str
    flow_id: Optional[str] = None
    reply_to: Optional[str] = None
    broadcast: bool = False
    hop_count: int = 0
    max_hops: int = 10
    ttl: int = 300  # Time to live in seconds


class Trace(BaseModel):
    """Tracing information for debugging and monitoring."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    start_time: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    agent_hops: List[Dict[str, str]] = Field(default_factory=list)


class MessageHeader(BaseModel):
    """Standard header for all messages in the MCP."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    route: Route
    trace: Trace = Field(default_factory=Trace)
    correlation_id: Optional[str] = None
    version: str = "1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Standard message format for the MCP."""
    header: MessageHeader
    body: Dict[str, Any] = Field(default_factory=dict)


# Specialized message body models
class ComputationRequestBody(BaseModel):
    """Body for computation request messages."""
    operation: str  # e.g., "derivative", "integrate", "solve"
    expression: str  # LaTeX or string representation of the expression
    variables: Optional[List[str]] = None
    domain: Optional[str] = None  # e.g., "calculus", "linear_algebra"
    step_by_step: bool = False
    format: str = "latex"  # Output format


class OCRRequestBody(BaseModel):
    """Body for OCR request messages."""
    image_path: str
    image_type: str = "handwritten_math"
    confidence_threshold: float = 0.7
    detect_diagrams: bool = True
    enhance_quality: bool = True


class VisualizationRequestBody(BaseModel):
    """Body for visualization request messages."""
    visualization_type: str  # e.g., "function_plot_2d", "function_plot_3d"
    data: Dict[str, Any]  # Visualization-specific data
    parameters: Dict[str, Any] = Field(default_factory=dict)  # Style, ranges, etc.
    format: str = "png"  # Output format


# Factory functions for creating messages
def create_message(
    message_type: MessageType,
    sender: str,
    recipient: str,
    body: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL,
    conversation_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    reply_to: Optional[str] = None,
    broadcast: bool = False,
    flow_id: Optional[str] = None,
    parent_trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Message:
    """Create a standardized message."""
    route = Route(
        sender=sender,
        recipient=recipient,
        flow_id=flow_id or str(uuid.uuid4()),
        reply_to=reply_to,
        broadcast=broadcast
    )
    
    trace = Trace()
    if parent_trace_id:
        trace.trace_id = parent_trace_id
        trace.parent_span_id = parent_span_id
    
    header = MessageHeader(
        message_type=message_type,
        priority=priority,
        route=route,
        trace=trace,
        conversation_id=conversation_id,
        correlation_id=correlation_id or str(uuid.uuid4())
    )
    
    return Message(header=header, body=body)


def create_computation_request(
    sender: str,
    expression: str,
    operation: str,
    variables: Optional[List[str]] = None,
    domain: Optional[str] = None,
    step_by_step: bool = False,
    format: str = "latex",
    priority: MessagePriority = MessagePriority.NORMAL,
    **kwargs
) -> Message:
    """Create a computation request message."""
    body = ComputationRequestBody(
        expression=expression,
        operation=operation,
        variables=variables,
        domain=domain,
        step_by_step=step_by_step,
        format=format
    ).dict()
    
    return create_message(
        message_type=MessageType.COMPUTATION_REQUEST,
        sender=sender,
        recipient="math_computation_agent",
        body=body,
        priority=priority,
        **kwargs
    )


def create_visualization_request(
    sender: str,
    visualization_type: str,
    data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    format: str = "png",
    priority: MessagePriority = MessagePriority.NORMAL,
    **kwargs
) -> Message:
    """Create a visualization request message."""
    body = VisualizationRequestBody(
        visualization_type=visualization_type,
        data=data,
        parameters=parameters or {},
        format=format
    ).dict()
    
    return create_message(
        message_type=MessageType.VISUALIZATION_REQUEST,
        sender=sender,
        recipient="visualization_agent",
        body=body,
        priority=priority,
        **kwargs
    )


def create_error_response(
    sender: str,
    recipient: str,
    error_message: str,
    error_code: str,
    original_message_id: Optional[str] = None,
    **kwargs
) -> Message:
    """Create an error response message."""
    body = {
        "error_message": error_message,
        "error_code": error_code,
        "original_message_id": original_message_id
    }
    
    return create_message(
        message_type=MessageType.ERROR,
        sender=sender,
        recipient=recipient,
        body=body,
        priority=MessagePriority.HIGH,
        **kwargs
    )

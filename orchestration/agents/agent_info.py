"""
Agent Information Definitions for the Mathematical Multimodal LLM System.

This module provides structured definitions for agent capabilities and metadata.
"""
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
import datetime


class AgentCapability(BaseModel):
    """Definition of an agent capability."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    agent_types: List[str] = Field(default_factory=list)
    domain: Optional[str] = None


class AgentMetadata(BaseModel):
    """Metadata about an agent."""
    name: str
    description: str
    version: str
    author: str
    license: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None
    additional_info: Dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    """Complete information about an agent."""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "registered"
    registered_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_seen: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    health_metrics: Dict[str, Any] = Field(default_factory=dict)


# Define standard capabilities for core agent types

LLM_AGENT_CAPABILITIES = [
    AgentCapability(
        name="classify_query",
        description="Classifies a mathematical query to determine domain and operations",
        parameters={
            "query": "Natural language query to classify",
            "classify_domain": "Boolean indicating whether to identify mathematical domain",
            "extract_expressions": "Boolean indicating whether to extract mathematical expressions"
        },
        examples=[
            {
                "query": "Find the derivative of x^2 sin(x)",
                "result": {
                    "domain": "calculus",
                    "operation": "differentiate",
                    "expressions": ["x^2 \\sin(x)"]
                }
            }
        ],
        agent_types=["llm"],
        domain="mathematics"
    ),
    AgentCapability(
        name="generate_response",
        description="Generates a natural language response to a mathematical query",
        parameters={
            "include_steps": "Boolean indicating whether to include step-by-step explanation",
            "response_type": "Type of response (explanation, concise, educational)",
            "format": "Output format (text, latex)"
        },
        examples=[],
        agent_types=["llm"],
        domain="mathematics"
    ),
    AgentCapability(
        name="translate_natural_language_to_latex",
        description="Converts a natural language mathematical expression to LaTeX",
        parameters={
            "text": "Natural language description of mathematical expression",
            "domain": "Optional mathematical domain for context"
        },
        examples=[
            {
                "text": "the integral of x squared times sine of x",
                "result": "\\int x^2 \\sin(x) \\, dx"
            }
        ],
        agent_types=["llm"],
        domain="mathematics"
    )
]

MATH_COMPUTATION_CAPABILITIES = [
    AgentCapability(
        name="compute",
        description="Performs a mathematical computation on an expression",
        parameters={
            "expression": "LaTeX or string representation of the expression",
            "operation": "Operation to perform (evaluate, simplify, etc.)",
            "domain": "Mathematical domain for context",
            "step_by_step": "Boolean indicating whether to generate steps"
        },
        examples=[],
        agent_types=["computation"],
        domain="mathematics"
    ),
    AgentCapability(
        name="solve_equation",
        description="Solves an equation for a specified variable",
        parameters={
            "equation": "Equation to solve",
            "variable": "Variable to solve for",
            "domain": "Mathematical domain for context"
        },
        examples=[],
        agent_types=["computation"],
        domain="algebra"
    ),
    AgentCapability(
        name="differentiate",
        description="Calculates the derivative of an expression",
        parameters={
            "expression": "Expression to differentiate",
            "variable": "Variable to differentiate with respect to",
            "order": "Order of differentiation"
        },
        examples=[],
        agent_types=["computation"],
        domain="calculus"
    ),
    AgentCapability(
        name="integrate",
        description="Calculates the integral of an expression",
        parameters={
            "expression": "Expression to integrate",
            "variable": "Variable to integrate with respect to",
            "lower_bound": "Optional lower bound for definite integrals",
            "upper_bound": "Optional upper bound for definite integrals"
        },
        examples=[],
        agent_types=["computation"],
        domain="calculus"
    )
]

OCR_CAPABILITIES = [
    AgentCapability(
        name="recognize_math",
        description="Recognizes mathematical notation from an image",
        parameters={
            "image_path": "Path to the image file",
            "detect_diagrams": "Boolean indicating whether to detect diagrams",
            "enhance_quality": "Boolean indicating whether to enhance image quality"
        },
        examples=[],
        agent_types=["ocr"],
        domain="mathematics"
    ),
    AgentCapability(
        name="extract_diagram",
        description="Extracts and classifies mathematical diagrams from an image",
        parameters={
            "image_path": "Path to the image file",
            "diagram_types": "List of diagram types to detect"
        },
        examples=[],
        agent_types=["ocr"],
        domain="mathematics"
    )
]

VISUALIZATION_CAPABILITIES = [
    AgentCapability(
        name="generate_visualization",
        description="Generates a visualization for a mathematical expression or data",
        parameters={
            "visualization_type": "Type of visualization to generate",
            "expression": "Mathematical expression to visualize",
            "domain": "Mathematical domain for context",
            "format": "Output format (png, svg, etc.)"
        },
        examples=[],
        agent_types=["visualization"],
        domain="mathematics"
    ),
    AgentCapability(
        name="plot_function",
        description="Plots a mathematical function in 2D",
        parameters={
            "expression": "Function expression",
            "x_range": "Range of x values",
            "y_range": "Optional range of y values",
            "features": "Features to highlight (critical points, etc.)"
        },
        examples=[],
        agent_types=["visualization"],
        domain="mathematics"
    ),
    AgentCapability(
        name="plot_3d",
        description="Creates a 3D plot of a function or data",
        parameters={
            "expression": "Function expression",
            "x_range": "Range of x values",
            "y_range": "Range of y values",
            "z_range": "Optional range of z values"
        },
        examples=[],
        agent_types=["visualization"],
        domain="mathematics"
    )
]

SEARCH_CAPABILITIES = [
    AgentCapability(
        name="external_search",
        description="Searches external sources for mathematical information",
        parameters={
            "query": "Search query",
            "sources": "List of sources to search",
            "max_results": "Maximum number of results to return"
        },
        examples=[],
        agent_types=["search"],
        domain="mathematics"
    ),
    AgentCapability(
        name="knowledge_retrieval",
        description="Retrieves specific mathematical knowledge from external sources",
        parameters={
            "concept": "Mathematical concept to retrieve",
            "domain": "Mathematical domain for context",
            "detail_level": "Level of detail required"
        },
        examples=[],
        agent_types=["search"],
        domain="mathematics"
    )
]


# All capabilities
ALL_CAPABILITIES = (
    LLM_AGENT_CAPABILITIES +
    MATH_COMPUTATION_CAPABILITIES +
    OCR_CAPABILITIES +
    VISUALIZATION_CAPABILITIES +
    SEARCH_CAPABILITIES
)

# Capability lookup by name
CAPABILITY_BY_NAME = {cap.name: cap for cap in ALL_CAPABILITIES}

def get_capability_info(capability_name: str) -> Optional[AgentCapability]:
    """Get information about a capability by name."""
    return CAPABILITY_BY_NAME.get(capability_name)


# Agent type definitions
AGENT_TYPES = {
    "llm": {
        "description": "Language model agent for natural language understanding and generation",
        "capabilities": [cap.name for cap in LLM_AGENT_CAPABILITIES]
    },
    "computation": {
        "description": "Mathematical computation agent for symbolic and numerical calculations",
        "capabilities": [cap.name for cap in MATH_COMPUTATION_CAPABILITIES]
    },
    "ocr": {
        "description": "Optical character recognition agent for mathematical notation",
        "capabilities": [cap.name for cap in OCR_CAPABILITIES]
    },
    "visualization": {
        "description": "Visualization agent for mathematical plots and diagrams",
        "capabilities": [cap.name for cap in VISUALIZATION_CAPABILITIES]
    },
    "search": {
        "description": "Search agent for retrieving external mathematical information",
        "capabilities": [cap.name for cap in SEARCH_CAPABILITIES]
    }
}

def get_agent_type_info(agent_type: str) -> Optional[Dict[str, Any]]:
    """Get information about an agent type."""
    return AGENT_TYPES.get(agent_type)

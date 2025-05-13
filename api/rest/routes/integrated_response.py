"""
API routes for integrated response generation
"""

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
import logging

from math_processing.formatting.latex_formatter import LatexFormatter
from math_processing.formatting.response_formatter import ResponseFormatter

router = APIRouter(prefix="/responses", tags=["responses"])

# Initialize formatters
logger = logging.getLogger(__name__)
latex_formatter = LatexFormatter()
response_formatter = ResponseFormatter(latex_formatter)


class IntegratedResponseRequest(BaseModel):
    """Request model for generating an integrated response."""
    explanation: str
    latex_expressions: Optional[Union[List[str], str]] = None
    steps: Optional[List[Dict[str, Any]]] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    citations: Optional[List[str]] = None
    domain: Optional[str] = None
    format_type: Optional[str] = "default"
    complexity_level: Optional[str] = "auto"
    include_citations: Optional[bool] = True


class IntegratedResponseResponse(BaseModel):
    """Response model for integrated response."""
    formatted_response: Dict[str, Any]


@router.post("/generate", response_model=IntegratedResponseResponse)
async def generate_integrated_response(request: IntegratedResponseRequest):
    """
    Generate an integrated response with properly formatted content.
    """
    try:
        # Prepare response data dictionary
        response_data = {
            "explanation": request.explanation,
            "latex_expressions": request.latex_expressions or [],
            "steps": request.steps or [],
            "visualizations": request.visualizations or [],
            "citations": request.citations or [],
            "domain": request.domain
        }
        
        # Format the response using the ResponseFormatter
        formatted_response = response_formatter.format_response(
            response_data=response_data,
            format_type=request.format_type,
            complexity_level=request.complexity_level,
            include_citations=request.include_citations
        )
        
        return IntegratedResponseResponse(
            formatted_response=formatted_response
        )
        
    except Exception as e:
        logger.error(f"Error generating integrated response: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate integrated response: {str(e)}"
        )


@router.get("/format-types")
async def get_format_types():
    """
    Get available response format types.
    """
    return {
        "format_types": [
            {
                "id": "default",
                "name": "Default",
                "description": "Balanced presentation with clear explanations and appropriate visualizations."
            },
            {
                "id": "educational",
                "name": "Educational",
                "description": "Detailed explanations with learning objectives and conceptual breakdowns."
            },
            {
                "id": "concise",
                "name": "Concise",
                "description": "Brief, to-the-point explanations with essential information only."
            },
            {
                "id": "technical",
                "name": "Technical",
                "description": "Formal mathematical presentation with rigorous notation and theoretical context."
            }
        ]
    }


@router.get("/complexity-levels")
async def get_complexity_levels():
    """
    Get available complexity levels for responses.
    """
    return {
        "complexity_levels": [
            {
                "id": "auto",
                "name": "Auto-detect",
                "description": "Automatically determine the appropriate level based on content."
            },
            {
                "id": "basic",
                "name": "Basic",
                "description": "Simplified explanations suitable for beginners."
            },
            {
                "id": "intermediate",
                "name": "Intermediate",
                "description": "Balanced explanations for those with some mathematical background."
            },
            {
                "id": "advanced",
                "name": "Advanced",
                "description": "Detailed explanations using formal mathematical concepts and terminology."
            }
        ]
    }


@router.post("/preview-formatting")
async def preview_response_formatting(
    explanation: str = Body(..., embed=True),
    format_type: str = Body("default", embed=True),
    complexity_level: str = Body("auto", embed=True)
):
    """
    Preview how an explanation would be formatted in different styles.
    """
    try:
        # Create a simple response data with just the explanation
        response_data = {
            "explanation": explanation,
            "latex_expressions": [],
            "steps": [],
            "visualizations": [],
            "citations": []
        }
        
        # Format the response using the ResponseFormatter
        formatted_response = response_formatter.format_response(
            response_data=response_data,
            format_type=format_type,
            complexity_level=complexity_level,
            include_citations=False
        )
        
        # Return just the formatted explanation
        return {
            "formatted_explanation": formatted_response["explanation"],
            "format_type": format_type,
            "complexity_level": formatted_response["complexity_level"]  # This might be detected if auto
        }
        
    except Exception as e:
        logger.error(f"Error previewing response formatting: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to preview response formatting: {str(e)}"
        )

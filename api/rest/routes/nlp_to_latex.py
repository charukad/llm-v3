"""
API routes for converting natural language to LaTeX
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
import logging

from math_processing.formatting.latex_formatter import LatexFormatter
from core.agent.llm_agent import CoreLLMAgent

router = APIRouter(prefix="/math", tags=["math"])

# Initialize components
logger = logging.getLogger(__name__)
latex_formatter = LatexFormatter()
llm_agent = CoreLLMAgent()

class NaturalToLatexRequest(BaseModel):
    """Request model for converting natural language to LaTeX."""
    natural_text: str
    context: Optional[Dict[str, Any]] = None


class NaturalToLatexResponse(BaseModel):
    """Response model for LaTeX conversion."""
    latex: str
    confidence: float
    alternatives: Optional[List[str]] = None


@router.post("/natural-to-latex", response_model=NaturalToLatexResponse)
async def convert_natural_to_latex(request: NaturalToLatexRequest):
    """
    Convert natural language mathematical description to LaTeX.
    """
    try:
        # Get the natural language text
        natural_text = request.natural_text
        
        if not natural_text or natural_text.strip() == "":
            raise HTTPException(status_code=400, detail="Natural text cannot be empty")
        
        # Use the LLM to convert natural language to LaTeX
        response = llm_agent.process_math_nl_to_latex(
            natural_text=natural_text,
            context=request.context
        )
        
        # Format the LaTeX for consistent style
        formatted_latex = latex_formatter.format_expression(response.get("latex", ""))
        
        # Return the formatted LaTeX
        return NaturalToLatexResponse(
            latex=formatted_latex,
            confidence=response.get("confidence", 0.9),
            alternatives=response.get("alternatives")
        )
        
    except Exception as e:
        logger.error(f"Error converting natural language to LaTeX: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to convert natural language to LaTeX: {str(e)}"
        )


class LatexToNaturalRequest(BaseModel):
    """Request model for converting LaTeX to natural language."""
    latex: str
    verbosity: Optional[str] = "normal"  # "brief", "normal", "detailed"


class LatexToNaturalResponse(BaseModel):
    """Response model for natural language conversion."""
    natural_text: str


@router.post("/latex-to-natural", response_model=LatexToNaturalResponse)
async def convert_latex_to_natural(request: LatexToNaturalRequest):
    """
    Convert LaTeX to natural language description.
    """
    try:
        # Get the LaTeX
        latex = request.latex
        
        if not latex or latex.strip() == "":
            raise HTTPException(status_code=400, detail="LaTeX cannot be empty")
        
        # Use the LLM to convert LaTeX to natural language
        response = llm_agent.process_math_latex_to_nl(
            latex=latex,
            verbosity=request.verbosity
        )
        
        # Return the natural language description
        return LatexToNaturalResponse(
            natural_text=response.get("natural_text", "")
        )
        
    except Exception as e:
        logger.error(f"Error converting LaTeX to natural language: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to convert LaTeX to natural language: {str(e)}"
        )

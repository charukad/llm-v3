"""
API routes for LaTeX formatting and processing
"""

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any

from math_processing.formatting.latex_formatter import LatexFormatter

router = APIRouter(prefix="/latex", tags=["latex"])

# Initialize formatter
latex_formatter = LatexFormatter()


class FormatExpressionRequest(BaseModel):
    """Request model for formatting a LaTeX expression."""
    expression: str
    display_context: Optional[str] = "web"  # web, print, presentation
    domain: Optional[str] = None  # calculus, linear_algebra, statistics, etc.


class FormatExpressionResponse(BaseModel):
    """Response model for formatted LaTeX expression."""
    formatted_expression: str
    display_context: str
    domain: Optional[str] = None


class FormatStepSolutionRequest(BaseModel):
    """Request model for formatting a step-by-step solution."""
    steps: List[Dict[str, Any]]
    display_context: Optional[str] = "web"


class FormatStepSolutionResponse(BaseModel):
    """Response model for formatted step-by-step solution."""
    formatted_steps: List[Dict[str, Any]]
    display_context: str


@router.post("/format-expression", response_model=FormatExpressionResponse)
async def format_expression(request: FormatExpressionRequest):
    """
    Format a LaTeX expression with enhanced typographical features.
    """
    try:
        # Format the expression
        formatted_expr = latex_formatter.format_expression(request.expression)
        
        # Apply context-specific optimizations
        if request.display_context:
            formatted_expr = latex_formatter.optimize_for_display(
                formatted_expr, request.display_context
            )
        
        # Apply domain-specific formatting if provided
        if request.domain:
            formatted_expr = latex_formatter._apply_domain_formatting(
                formatted_expr, request.domain
            )
        
        return FormatExpressionResponse(
            formatted_expression=formatted_expr,
            display_context=request.display_context,
            domain=request.domain
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error formatting expression: {str(e)}")


@router.post("/format-solution", response_model=FormatStepSolutionResponse)
async def format_step_solution(request: FormatStepSolutionRequest):
    """
    Format a step-by-step solution with consistent LaTeX formatting.
    """
    try:
        # Format the step solution
        formatted_steps = latex_formatter.format_step_solution(request.steps)
        
        # Apply display context optimizations to each step
        if request.display_context:
            for step in formatted_steps:
                if "input" in step and step["input"]:
                    step["input"] = latex_formatter.optimize_for_display(
                        step["input"], request.display_context
                    )
                if "output" in step and step["output"]:
                    step["output"] = latex_formatter.optimize_for_display(
                        step["output"], request.display_context
                    )
        
        return FormatStepSolutionResponse(
            formatted_steps=formatted_steps,
            display_context=request.display_context
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error formatting solution: {str(e)}")


@router.post("/create-equation-environment")
async def create_equation_environment(
    expressions: List[str] = Body(..., embed=True),
    numbered: bool = Body(False, embed=True),
    aligned: bool = Body(False, embed=True)
):
    """
    Create a proper LaTeX equation environment.
    """
    try:
        result = latex_formatter.create_equation_environment(
            expressions=expressions,
            numbered=numbered,
            aligned=aligned
        )
        return {"latex": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating equation environment: {str(e)}")


@router.post("/format-matrix")
async def format_matrix(
    matrix: List[List[Any]] = Body(..., embed=True),
    bracket_type: str = Body("bracket", embed=True),
    alignment: str = Body("c", embed=True)
):
    """
    Format a matrix with proper LaTeX conventions.
    """
    try:
        result = latex_formatter.format_matrix(
            matrix=matrix,
            bracket_type=bracket_type,
            alignment=alignment
        )
        return {"latex": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error formatting matrix: {str(e)}")

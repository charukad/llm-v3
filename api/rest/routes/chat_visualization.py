"""
Chat Visualization Router - Process text to generate visualizations.

This module handles the processing of natural language requests for visualizations.
"""
from typing import Optional, Dict, Any, List
import logging
import uuid
import os
import json

from fastapi import APIRouter, Body, HTTPException, Query, Depends, Request, File, UploadFile, Form
from pydantic import BaseModel, Field

from orchestration.agents.chat_analysis_agent import get_chat_analysis_agent
from api.rest.routes.visualization import viz_agent

# Import visualization endpoints to call directly
from api.rest.routes.visualization import (
    plot_function, plot_multiple_functions, plot_3d_surface, plot_parametric_3d
)

router = APIRouter(prefix="/chat")
logger = logging.getLogger(__name__)

# Initialize the chat analysis agent
chat_analysis_agent = get_chat_analysis_agent()

class ChatVisualizationRequest(BaseModel):
    """Chat visualization request model."""
    text: str = Field(..., description="The text query to analyze")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    generate_immediately: bool = Field(True, description="Whether to generate the visualization immediately if possible")

@router.post("/visualize")
async def process_visualization_request(request: ChatVisualizationRequest):
    """
    Process a natural language request for visualization.
    
    This endpoint analyzes the text to identify if it's requesting a visualization,
    extracts the necessary parameters, and generates the appropriate plot.
    """
    try:
        # Analyze the text for visualization requests
        analysis_result = chat_analysis_agent.analyze_plot_request(request.text)
        
        # If not a visualization request, return the analysis
        if not analysis_result.get("is_visualization_request", False):
            return {
                "success": True,
                "is_visualization_request": False,
                "message": "The text does not appear to be a visualization request.",
                "analysis": analysis_result
            }
        
        # If we don't want to generate immediately, just return the analysis
        if not request.generate_immediately:
            return {
                "success": True,
                "is_visualization_request": True,
                "plot_type": analysis_result.get("plot_type"),
                "parameters": analysis_result.get("parameters", {}),
                "message": "Visualization parameters extracted successfully. Set generate_immediately=true to create the visualization.",
                "analysis": analysis_result
            }
        
        # Process the visualization request
        plot_type = analysis_result.get("plot_type")
        parameters = analysis_result.get("parameters", {})
        
        visualization_result = None
        
        # Call the appropriate visualization endpoint based on the plot type
        if plot_type == "function_2d":
            # Convert parameters to the expected format
            x_min = parameters.get("x_range", [-10, 10])[0] if isinstance(parameters.get("x_range"), list) else -10
            x_max = parameters.get("x_range", [-10, 10])[1] if isinstance(parameters.get("x_range"), list) else 10
            
            visualization_result = await plot_function(
                expression=parameters.get("expression"),
                x_min=x_min,
                x_max=x_max,
                title=parameters.get("title"),
                x_label=parameters.get("x_label", "x"),
                y_label=parameters.get("y_label", "y"),
                num_points=parameters.get("num_points", 1000),
                show_grid=parameters.get("show_grid", True),
            )
        
        elif plot_type == "functions_2d":
            # Convert parameters to the expected format
            x_min = parameters.get("x_range", [-10, 10])[0] if isinstance(parameters.get("x_range"), list) else -10
            x_max = parameters.get("x_range", [-10, 10])[1] if isinstance(parameters.get("x_range"), list) else 10
            
            visualization_result = await plot_multiple_functions(
                expressions=parameters.get("expressions"),
                labels=parameters.get("labels"),
                x_min=x_min,
                x_max=x_max,
                title=parameters.get("title", "Multiple Functions"),
                x_label=parameters.get("x_label", "x"),
                y_label=parameters.get("y_label", "y"),
                num_points=parameters.get("num_points", 1000),
                show_grid=parameters.get("show_grid", True),
            )
        
        elif plot_type == "function_3d":
            # Convert parameters to the expected format
            x_min = parameters.get("x_range", [-5, 5])[0] if isinstance(parameters.get("x_range"), list) else -5
            x_max = parameters.get("x_range", [-5, 5])[1] if isinstance(parameters.get("x_range"), list) else 5
            y_min = parameters.get("y_range", [-5, 5])[0] if isinstance(parameters.get("y_range"), list) else -5
            y_max = parameters.get("y_range", [-5, 5])[1] if isinstance(parameters.get("y_range"), list) else 5
            
            visualization_result = await plot_3d_surface(
                expression=parameters.get("expression"),
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                title=parameters.get("title"),
                x_label=parameters.get("x_label", "x"),
                y_label=parameters.get("y_label", "y"),
                z_label=parameters.get("z_label", "z"),
                num_points=parameters.get("num_points", 50),
                cmap=parameters.get("cmap", "viridis"),
                view_angle=parameters.get("view_angle", [30, 30]),
            )
        
        elif plot_type == "parametric_3d":
            # Convert parameters to the expected format
            t_min = parameters.get("t_range", [0, 6.28])[0] if isinstance(parameters.get("t_range"), list) else 0
            t_max = parameters.get("t_range", [0, 6.28])[1] if isinstance(parameters.get("t_range"), list) else 6.28
            
            visualization_result = await plot_parametric_3d(
                x_expression=parameters.get("x_expression"),
                y_expression=parameters.get("y_expression"),
                z_expression=parameters.get("z_expression"),
                t_min=t_min,
                t_max=t_max,
                title=parameters.get("title"),
                x_label=parameters.get("x_label", "x"),
                y_label=parameters.get("y_label", "y"),
                z_label=parameters.get("z_label", "z"),
                num_points=parameters.get("num_points", 1000),
                color=parameters.get("color", "blue"),
                view_angle=parameters.get("view_angle", [30, 30]),
            )
        
        else:
            return {
                "success": False,
                "error": f"Unsupported plot type: {plot_type}",
                "is_visualization_request": True,
                "plot_type": plot_type,
                "parameters": parameters
            }
        
        # Return the combined result
        return {
            "success": visualization_result.get("success", False),
            "is_visualization_request": True,
            "plot_type": plot_type,
            "parameters": parameters,
            "visualization": visualization_result,
            "message": "Visualization created successfully" if visualization_result.get("success", False) else "Failed to create visualization",
            "analysis": analysis_result
        }
        
    except Exception as e:
        logger.error(f"Error processing visualization request: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing visualization request: {str(e)}")

# Integration with workflow
@router.post("/process")
async def process_chat_to_visualization(
    text: str = Body(..., embed=True),
    conversation_id: Optional[str] = Body(None, embed=True),
):
    """
    Process a chat message and determine if it should be routed to visualization.
    
    This endpoint is suitable for integration with the workflow system.
    """
    try:
        # Create a request object
        request = ChatVisualizationRequest(
            text=text,
            conversation_id=conversation_id,
            generate_immediately=True
        )
        
        # Process the request
        result = await process_visualization_request(request)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing chat to visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat to visualization: {str(e)}") 
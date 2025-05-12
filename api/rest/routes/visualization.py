from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Form, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List, Optional
import os
import json
import uuid

from visualization.agent.viz_agent import VisualizationAgent
from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent
from visualization.selection.context_analyzer import VisualizationSelector
from database.access.visualization_repository import VisualizationRepository

# Initialize router
router = APIRouter(prefix="/visualization", tags=["visualization"])

# Initialize agents
VISUALIZATION_DIR = os.environ.get("VISUALIZATION_DIR", "visualizations")
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

base_config = {
    "storage_dir": VISUALIZATION_DIR,
    "use_database": True
}

viz_agent = VisualizationAgent(base_config)
advanced_viz_agent = AdvancedVisualizationAgent(base_config)
viz_selector = VisualizationSelector()
viz_repository = VisualizationRepository()

@router.post("/generate")
async def generate_visualization(
    visualization_type: str = Form(...),
    params: str = Form(...),
    interaction_id: Optional[str] = Form(None)
):
    """Generate a visualization with the specified parameters."""
    try:
        # Parse parameters
        parameters = json.loads(params)
        
        # Create message format expected by agent
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "api",
                "recipient": "visualization_agent",
                "timestamp": "",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": visualization_type,
                "parameters": parameters
            }
        }
        
        # Determine which agent to use based on visualization type
        agent = viz_agent
        advanced_types = advanced_viz_agent.get_capabilities().get("advanced_features", [])
        
        if visualization_type in advanced_types:
            agent = advanced_viz_agent
        
        # Process the message
        result = agent.process_message(message)
        
        # Add interaction ID if provided and storing to database
        if interaction_id and "file_path" in result and result.get("success", False):
            try:
                viz_id = viz_repository.store_visualization(
                    visualization_type=visualization_type,
                    parameters=parameters,
                    file_path=result["file_path"],
                    metadata=result.get("data", {}),
                    interaction_id=interaction_id
                )
                result["visualization_id"] = viz_id
            except Exception as e:
                result["warning"] = f"Failed to store visualization in database: {str(e)}"
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {str(e)}")

@router.post("/select")
async def select_visualization(
    context: Dict[str, Any] = Body(...)
):
    """Select appropriate visualization based on mathematical context."""
    try:
        # Process with selector
        result = viz_selector.select_visualization(context)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to select visualization: {str(e)}")

@router.get("/types")
async def get_visualization_types():
    """Get all supported visualization types."""
    try:
        # Get base and advanced types
        base_capabilities = viz_agent.get_capabilities()
        advanced_capabilities = advanced_viz_agent.get_capabilities()
        
        return {
            "base_types": base_capabilities.get("supported_types", []),
            "advanced_types": advanced_capabilities.get("supported_types", []),
            "advanced_features": advanced_capabilities.get("advanced_features", [])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get visualization types: {str(e)}")

@router.get("/by-id/{visualization_id}")
async def get_visualization_by_id(
    visualization_id: str,
    as_file: bool = Query(False)
):
    """Get a visualization by ID."""
    try:
        # Retrieve from database
        visualization = viz_repository.get_visualization(visualization_id)
        
        if not visualization:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        # Check if file exists
        if "file_path" in visualization and os.path.exists(visualization["file_path"]):
            # Return file or metadata based on parameter
            if as_file:
                return FileResponse(
                    path=visualization["file_path"],
                    filename=os.path.basename(visualization["file_path"]),
                    media_type="image/png"
                )
            else:
                return visualization
        else:
            raise HTTPException(status_code=404, detail="Visualization file not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve visualization: {str(e)}")

@router.get("/by-interaction/{interaction_id}")
async def get_visualizations_by_interaction(
    interaction_id: str
):
    """Get all visualizations for an interaction."""
    try:
        # Retrieve from database
        visualizations = viz_repository.get_visualizations_by_interaction(interaction_id)
        
        return {
            "interaction_id": interaction_id,
            "visualizations": visualizations,
            "count": len(visualizations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve visualizations: {str(e)}")

@router.get("/recent")
async def get_recent_visualizations(
    limit: int = Query(10, ge=1, le=100)
):
    """Get recent visualizations."""
    try:
        # Retrieve from database
        visualizations = viz_repository.get_recent_visualizations(limit)
        
        return {
            "visualizations": visualizations,
            "count": len(visualizations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recent visualizations: {str(e)}")

@router.get("/file/{filename}")
async def get_visualization_by_filename(
    filename: str
):
    """Get a visualization file directly by filename."""
    try:
        # Construct the full path
        file_path = os.path.join(VISUALIZATION_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Visualization file {filename} not found")
        
        # Return the file
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="image/png"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve visualization file: {str(e)}")

@router.get("/files")
async def list_visualization_files():
    """List all visualization files in the visualization directory."""
    try:
        # List all files in the visualization directory
        files = os.listdir(VISUALIZATION_DIR)
        
        # Filter out non-files (e.g., directories)
        files = [f for f in files if os.path.isfile(os.path.join(VISUALIZATION_DIR, f))]
        
        # Add full paths and creation times
        file_info = []
        for filename in files:
            path = os.path.join(VISUALIZATION_DIR, filename)
            file_info.append({
                "filename": filename,
                "full_path": path,
                "created": os.path.getctime(path),
                "size": os.path.getsize(path)
            })
            
        # Sort by creation time (newest first)
        file_info.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "visualization_dir": VISUALIZATION_DIR,
            "files": file_info,
            "count": len(file_info)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list visualization files: {str(e)}")

@router.post("/plot/function")
async def plot_function(
    expression: str = Body(..., description="Mathematical expression to plot (e.g., 'sin(x)')"),
    x_min: float = Body(-10, description="Minimum x value"),
    x_max: float = Body(10, description="Maximum x value"),
    title: Optional[str] = Body(None, description="Plot title"),
    x_label: str = Body("x", description="X-axis label"),
    y_label: str = Body("y", description="Y-axis label"),
    num_points: int = Body(1000, description="Number of points to sample"),
    show_grid: bool = Body(True, description="Whether to show grid lines"),
    interaction_id: Optional[str] = Body(None, description="Interaction ID for tracking")
):
    """Plot a mathematical function with a user-friendly interface."""
    try:
        # Prepare parameters
        parameters = {
            "expression": expression,
            "x_range": [x_min, x_max],
            "title": title or f"f(x) = {expression}",
            "x_label": x_label,
            "y_label": y_label,
            "num_points": num_points,
            "show_grid": show_grid
        }
        
        # Create message
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "api",
                "recipient": "visualization_agent",
                "timestamp": "",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": "function_2d",
                "parameters": parameters
            }
        }
        
        # Process with the visualization agent
        result = viz_agent.process_message(message)
        
        # If successful and we have a file path, add direct access URL
        if result.get("success", False) and "file_path" in result:
            file_path = result["file_path"]
            filename = os.path.basename(file_path)
            result["url"] = f"/visualization/file/{filename}"
            
            # Store in database if interaction_id provided
            if interaction_id:
                try:
                    viz_id = viz_repository.store_visualization(
                        visualization_type="function_2d",
                        parameters=parameters,
                        file_path=file_path,
                        metadata=result.get("data", {}),
                        interaction_id=interaction_id
                    )
                    result["visualization_id"] = viz_id
                except Exception as e:
                    result["warning"] = f"Failed to store visualization in database: {str(e)}"
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to plot function: {str(e)}")
        
@router.post("/plot/functions")
async def plot_multiple_functions(
    expressions: List[str] = Body(..., description="List of mathematical expressions to plot"),
    labels: Optional[List[str]] = Body(None, description="Labels for each function"),
    x_min: float = Body(-10, description="Minimum x value"),
    x_max: float = Body(10, description="Maximum x value"),
    title: str = Body("Multiple Functions", description="Plot title"),
    x_label: str = Body("x", description="X-axis label"),
    y_label: str = Body("y", description="Y-axis label"),
    num_points: int = Body(1000, description="Number of points to sample"),
    show_grid: bool = Body(True, description="Whether to show grid lines"),
    interaction_id: Optional[str] = Body(None, description="Interaction ID for tracking")
):
    """Plot multiple mathematical functions on the same graph."""
    try:
        # Prepare parameters
        parameters = {
            "expressions": expressions,
            "labels": labels,
            "x_range": [x_min, x_max],
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "num_points": num_points,
            "show_grid": show_grid
        }
        
        # Create message
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "api",
                "recipient": "visualization_agent",
                "timestamp": "",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": "functions_2d",
                "parameters": parameters
            }
        }
        
        # Process with the visualization agent
        result = viz_agent.process_message(message)
        
        # If successful and we have a file path, add direct access URL
        if result.get("success", False) and "file_path" in result:
            file_path = result["file_path"]
            filename = os.path.basename(file_path)
            result["url"] = f"/visualization/file/{filename}"
            
            # Store in database if interaction_id provided
            if interaction_id:
                try:
                    viz_id = viz_repository.store_visualization(
                        visualization_type="functions_2d",
                        parameters=parameters,
                        file_path=file_path,
                        metadata=result.get("data", {}),
                        interaction_id=interaction_id
                    )
                    result["visualization_id"] = viz_id
                except Exception as e:
                    result["warning"] = f"Failed to store visualization in database: {str(e)}"
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to plot functions: {str(e)}")

@router.post("/plot/surface3d")
async def plot_3d_surface(
    expression: str = Body(..., description="Mathematical expression to plot (e.g., 'sin(x)*cos(y)')"),
    x_min: float = Body(-5, description="Minimum x value"),
    x_max: float = Body(5, description="Maximum x value"),
    y_min: float = Body(-5, description="Minimum y value"),
    y_max: float = Body(5, description="Maximum y value"),
    title: Optional[str] = Body(None, description="Plot title"),
    x_label: str = Body("x", description="X-axis label"),
    y_label: str = Body("y", description="Y-axis label"),
    z_label: str = Body("z", description="Z-axis label"),
    num_points: int = Body(50, description="Number of points to sample (per axis)"),
    cmap: str = Body("viridis", description="Colormap for the surface"),
    view_angle: List[float] = Body([30, 30], description="Viewing angle [elevation, azimuth]"),
    interaction_id: Optional[str] = Body(None, description="Interaction ID for tracking")
):
    """Plot a 3D surface for a function of two variables z = f(x,y)."""
    try:
        # Prepare parameters
        parameters = {
            "expression": expression,
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "title": title or f"Surface plot of {expression}",
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "num_points": num_points,
            "cmap": cmap,
            "view_angle": view_angle
        }
        
        # Create message
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "api",
                "recipient": "visualization_agent",
                "timestamp": "",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": "function_3d",
                "parameters": parameters
            }
        }
        
        # Process with the visualization agent
        result = viz_agent.process_message(message)
        
        # If successful and we have a file path, add direct access URL
        if result.get("success", False) and "file_path" in result:
            file_path = result["file_path"]
            filename = os.path.basename(file_path)
            result["url"] = f"/visualization/file/{filename}"
            
            # Store in database if interaction_id provided
            if interaction_id:
                try:
                    viz_id = viz_repository.store_visualization(
                        visualization_type="function_3d",
                        parameters=parameters,
                        file_path=file_path,
                        metadata=result.get("data", {}),
                        interaction_id=interaction_id
                    )
                    result["visualization_id"] = viz_id
                except Exception as e:
                    result["warning"] = f"Failed to store visualization in database: {str(e)}"
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to plot 3D surface: {str(e)}")

@router.post("/plot/parametric3d")
async def plot_parametric_3d(
    x_expression: str = Body(..., description="X component expression (function of t)"),
    y_expression: str = Body(..., description="Y component expression (function of t)"),
    z_expression: str = Body(..., description="Z component expression (function of t)"),
    t_min: float = Body(0, description="Minimum t value"),
    t_max: float = Body(6.28, description="Maximum t value (default: 2π)"),
    title: Optional[str] = Body(None, description="Plot title"),
    x_label: str = Body("x", description="X-axis label"),
    y_label: str = Body("y", description="Y-axis label"),
    z_label: str = Body("z", description="Z-axis label"),
    num_points: int = Body(1000, description="Number of points to sample"),
    color: str = Body("blue", description="Line color"),
    view_angle: List[float] = Body([30, 30], description="Viewing angle [elevation, azimuth]"),
    interaction_id: Optional[str] = Body(None, description="Interaction ID for tracking")
):
    """Plot a 3D parametric curve where (x,y,z) are functions of parameter t."""
    try:
        # Prepare parameters
        parameters = {
            "x_expression": x_expression,
            "y_expression": y_expression,
            "z_expression": z_expression,
            "t_range": [t_min, t_max],
            "title": title or f"Parametric 3D curve",
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "num_points": num_points,
            "color": color,
            "view_angle": view_angle
        }
        
        # Create message
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "api",
                "recipient": "visualization_agent",
                "timestamp": "",
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": "parametric_3d",
                "parameters": parameters
            }
        }
        
        # Process with the visualization agent
        result = viz_agent.process_message(message)
        
        # If successful and we have a file path, add direct access URL
        if result.get("success", False) and "file_path" in result:
            file_path = result["file_path"]
            filename = os.path.basename(file_path)
            result["url"] = f"/visualization/file/{filename}"
            
            # Store in database if interaction_id provided
            if interaction_id:
                try:
                    viz_id = viz_repository.store_visualization(
                        visualization_type="parametric_3d",
                        parameters=parameters,
                        file_path=file_path,
                        metadata=result.get("data", {}),
                        interaction_id=interaction_id
                    )
                    result["visualization_id"] = viz_id
                except Exception as e:
                    result["warning"] = f"Failed to store visualization in database: {str(e)}"
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to plot parametric 3D curve: {str(e)}")

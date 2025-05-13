from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Form, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List, Optional
import os
import json
import uuid
import numpy as np

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

# Add numpy type conversion function
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

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
        
        # Convert any numpy types in the result
        result = convert_numpy_types(result)
        
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

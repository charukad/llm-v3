"""
API routes for executing workflows and fetching their status
"""

from fastapi import APIRouter, Body, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
import logging
import asyncio

from orchestration.manager.orchestration_manager import OrchestrationManager
from orchestration.workflow.workflow_registry import WorkflowRegistry

router = APIRouter(prefix="/workflows", tags=["workflows"])

# Initialize components
logger = logging.getLogger(__name__)
workflow_registry = WorkflowRegistry()
orchestration_manager = OrchestrationManager(workflow_registry=workflow_registry)


class IntegratedResponseRequest(BaseModel):
    """Request model for executing an integrated response workflow."""
    query: str
    context: Optional[Dict[str, Any]] = None
    format_type: Optional[str] = "default"
    complexity_level: Optional[str] = "auto"
    include_visualizations: Optional[bool] = True
    include_step_by_step: Optional[bool] = True
    include_search: Optional[bool] = True


class IntegratedResponseResponse(BaseModel):
    """Response model for integrated response execution."""
    workflow_id: str
    status: str
    estimated_time: Optional[float] = None


@router.post("/integrated-response", response_model=IntegratedResponseResponse)
async def execute_integrated_response(
    background_tasks: BackgroundTasks,
    request: IntegratedResponseRequest
):
    """
    Execute an integrated response workflow.
    """
    try:
        # Start the workflow
        workflow_id = orchestration_manager.start_workflow(
            workflow_type="integrated_response",
            query=request.query,
            context=request.context,
            format_type=request.format_type,
            complexity_level=request.complexity_level,
            include_visualizations=request.include_visualizations,
            include_step_by_step=request.include_step_by_step,
            include_search=request.include_search
        )
        
        # Run the workflow in the background
        background_tasks.add_task(
            orchestration_manager.execute_workflow,
            workflow_id
        )
        
        # Estimate the time based on query complexity
        estimated_time = _estimate_execution_time(request.query)
        
        return IntegratedResponseResponse(
            workflow_id=workflow_id,
            status="in_progress",
            estimated_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"Error executing integrated response workflow: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to execute integrated response workflow: {str(e)}"
        )


@router.get("/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Get the status of a workflow.
    """
    try:
        status = orchestration_manager.get_workflow_status(workflow_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
            
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get workflow status: {str(e)}"
        )


@router.get("/result/{workflow_id}")
async def get_workflow_result(workflow_id: str):
    """
    Get the result of a completed workflow.
    """
    try:
        result = orchestration_manager.get_workflow_result(workflow_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Workflow result {workflow_id} not found")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow result: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get workflow result: {str(e)}"
        )


@router.get("/available")
async def get_available_workflows():
    """
    Get available workflow types.
    """
    try:
        available_workflows = workflow_registry.get_available_workflows()
        return {"workflows": available_workflows}
        
    except Exception as e:
        logger.error(f"Error getting available workflows: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get available workflows: {str(e)}"
        )


def _estimate_execution_time(query: str) -> float:
    """
    Estimate execution time based on query complexity.
    
    Args:
        query: User query
        
    Returns:
        Estimated execution time in seconds
    """
    # This is a simple heuristic - in a real system this would be more sophisticated
    base_time = 5.0  # Base time in seconds
    
    # Longer queries likely need more processing
    length_factor = min(len(query) / 50, 3)  # Cap at 3x
    
    # Check for indicators of complex operations
    complexity_factor = 1.0
    complex_terms = ["integral", "derivative", "matrix", "eigenvalue", "system of equations"]
    for term in complex_terms:
        if term.lower() in query.lower():
            complexity_factor += 0.5  # Add 0.5 for each complex term
    
    # Calculate total estimated time
    estimated_time = base_time * (1 + length_factor) * complexity_factor
    
    return estimated_time

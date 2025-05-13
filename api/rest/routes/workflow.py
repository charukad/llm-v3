"""
API routes for end-to-end workflows.

This module provides REST API endpoints for end-to-end workflow processing,
which combines all system components to handle mathematical queries.
"""
import logging
import base64
import tempfile
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body, BackgroundTasks, File, UploadFile, Form, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from orchestration.workflow.end_to_end_workflows import EndToEndWorkflowManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/workflow",
    tags=["workflow"],
    responses={404: {"description": "Not found"}},
)

# Get workflow manager instance - initialized lazily on first request
from orchestration.agents.registry import get_agent_registry
from ..system_init import initialize_system

# Shared registry reference
registry = get_agent_registry()

# Workflow manager will be initialized on first access
workflow_manager = None

# Define models
class WorkflowInput(BaseModel):
    """Workflow input model."""
    input_type: str = Field(..., description="Input type (text, image, multipart)")
    content: Any = Field(..., description="Content based on input type")
    content_type: Optional[str] = Field(None, description="Content type for text")
    context_id: Optional[str] = Field(None, description="Context ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")


# Dependency to check if workflow manager is initialized
def get_workflow_manager():
    """Get the workflow manager instance or raise an error if not initialized."""
    global workflow_manager
    
    # Initialize workflow manager if not already done
    if workflow_manager is None:
        try:
            # Make sure system is initialized first
            initialize_system()
            
            # Now create workflow manager with registry that has services
            workflow_manager = EndToEndWorkflowManager(registry)
            logger.info("Workflow manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize workflow manager: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Workflow manager initialization failed: {str(e)}"
            )
    
    return workflow_manager


@router.post("/process")
async def process_workflow(
    input_data: WorkflowInput,
    wf_manager: EndToEndWorkflowManager = Depends(get_workflow_manager)
):
    """
    Process an end-to-end workflow.
    
    This endpoint starts a workflow that processes the input through all relevant
    system components, from input processing to response generation.
    """
    try:
        # Start workflow
        result = wf_manager.start_workflow(
            input_data=input_data.dict(),
            context_id=input_data.context_id,
            conversation_id=input_data.conversation_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting workflow: {str(e)}")


@router.post("/process/text")
async def process_text_workflow(
    text: str = Body(..., embed=True),
    context_id: Optional[str] = Body(None, embed=True),
    conversation_id: Optional[str] = Body(None, embed=True),
    wf_manager: EndToEndWorkflowManager = Depends(get_workflow_manager)
):
    """
    Process a text-based workflow.
    
    This endpoint is a simplified version of the process endpoint specifically for text input.
    """
    try:
        # Create input data
        input_data = {
            "input_type": "text",
            "content": text,
            "content_type": "text/plain",
            "context_id": context_id,
            "conversation_id": conversation_id
        }
        
        # Start workflow
        result = wf_manager.start_workflow(
            input_data=input_data,
            context_id=context_id,
            conversation_id=conversation_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting text workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting text workflow: {str(e)}")


@router.post("/process/image")
async def process_image_workflow(
    file: UploadFile,
    context_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    wf_manager: EndToEndWorkflowManager = Depends(get_workflow_manager)
):
    """
    Process an image-based workflow.
    
    This endpoint is a simplified version of the process endpoint specifically for image input.
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp.write(file_content)
            temp_path = temp.name
        
        try:
            # Create input data
            input_data = {
                "input_type": "image",
                "content": temp_path,
                "content_type": file.content_type,
                "context_id": context_id,
                "conversation_id": conversation_id
            }
            
            # Start workflow
            result = wf_manager.start_workflow(
                input_data=input_data,
                context_id=context_id,
                conversation_id=conversation_id
            )
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error starting image workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting image workflow: {str(e)}")


@router.get("/status/{workflow_id}")
async def get_workflow_status(
    workflow_id: str,
    wf_manager: EndToEndWorkflowManager = Depends(get_workflow_manager)
):
    """
    Get the status of a workflow.
    
    This endpoint returns the current status of a workflow, including completed steps.
    """
    try:
        status = wf_manager.get_workflow_status(workflow_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting workflow status: {str(e)}")


@router.get("/result/{workflow_id}")
async def get_workflow_result(
    workflow_id: str,
    wf_manager: EndToEndWorkflowManager = Depends(get_workflow_manager)
):
    """
    Get the result of a completed workflow.
    
    This endpoint returns the result of a completed workflow, including the generated response.
    """
    try:
        result = wf_manager.get_workflow_result(workflow_id)
        
        if not result.get("success", False):
            if "error" in result and "not completed" in result["error"]:
                # Workflow still in progress
                return {
                    "success": False,
                    "workflow_id": workflow_id,
                    "state": result.get("state", "unknown"),
                    "message": "Workflow still in progress"
                }
            else:
                # Other error
                raise HTTPException(status_code=404, detail=result.get("error", "Workflow result not found"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting workflow result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting workflow result: {str(e)}")

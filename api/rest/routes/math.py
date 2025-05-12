"""
Mathematical operation routes for the REST API.

This module provides the API endpoints for mathematical operations.
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
import uuid
import os
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/math", tags=["mathematics"])

# Request and response models
class MathQueryRequest(BaseModel):
    """Request model for mathematical queries."""
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None

class MathResponse(BaseModel):
    """Response model for mathematical operations."""
    workflow_id: str
    status: str = "processing"
    message: str = "Query is being processed"

# Dependency for getting system components
# These would be initialized at application startup in a real implementation
async def get_orchestration_manager():
    """Get the orchestration manager instance."""
    # Placeholder for real implementation
    return {"start_workflow": lambda *args, **kwargs: str(uuid.uuid4())}

async def get_conversation_repository():
    """Get the conversation repository instance."""
    # Placeholder for real implementation
    return {
        "create_conversation": lambda *args, **kwargs: str(uuid.uuid4()),
        "add_interaction": lambda *args, **kwargs: str(uuid.uuid4())
    }

# Routes
@router.post("/query", response_model=MathResponse)
async def math_query(
    request: MathQueryRequest,
    background_tasks: BackgroundTasks,
    orchestration_manager=Depends(get_orchestration_manager),
    conversation_repo=Depends(get_conversation_repository)
):
    """
    Process a mathematical query.
    
    Args:
        request: Mathematical query request
        background_tasks: Background task manager
        orchestration_manager: Orchestration manager instance
        conversation_repo: Conversation repository instance
        
    Returns:
        Response with workflow ID for tracking
    """
    try:
        # Create conversation if needed
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = conversation_repo["create_conversation"](
                user_id=request.user_id or "anonymous",
                title=f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        # Add user interaction
        interaction_id = conversation_repo["add_interaction"](
            conversation_id=conversation_id,
            user_input={"text": request.query, "type": "text"},
            system_response={"status": "processing"}
        )
        
        # Register the workflow with our new workflow system
        try:
            # Try to use the workflow API directly
            workflow_response = requests.post(
                "http://localhost:8000/workflow/register",
                params={"query": request.query}
            )
            
            if workflow_response.status_code == 200:
                workflow_id = workflow_response.json()["workflow_id"]
            else:
                # Fallback to traditional workflow creation
                workflow_id = orchestration_manager["start_workflow"](
                    workflow_type="math_problem_solving",
                    initial_data={
                        "query": request.query,
                        "conversation_id": conversation_id,
                        "interaction_id": interaction_id,
                        "user_id": request.user_id
                    }
                )
        except Exception as e:
            logger.warning(f"Could not register with workflow API: {e}")
            # Fallback to traditional workflow creation
        workflow_id = orchestration_manager["start_workflow"](
            workflow_type="math_problem_solving",
            initial_data={
                "query": request.query,
                "conversation_id": conversation_id,
                "interaction_id": interaction_id,
                "user_id": request.user_id
            }
        )
        
        # Return response with workflow ID for tracking
        return MathResponse(
            workflow_id=workflow_id,
            status="processing",
            message="Your mathematical query is being processed"
        )
    
    except Exception as e:
        logger.error(f"Error processing math query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/handwritten", response_model=MathResponse)
async def handwritten_math(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    orchestration_manager=Depends(get_orchestration_manager),
    conversation_repo=Depends(get_conversation_repository),
    conversation_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Process a handwritten mathematical expression.
    
    Args:
        background_tasks: Background task manager
        file: Uploaded image file with handwritten math
        orchestration_manager: Orchestration manager instance
        conversation_repo: Conversation repository instance
        conversation_id: Optional conversation ID
        user_id: Optional user ID
        
    Returns:
        Response with workflow ID for tracking
    """
    try:
        # Save the uploaded file
        file_path = f"/tmp/{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        contents = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Create conversation if needed
        if not conversation_id:
            conversation_id = conversation_repo["create_conversation"](
                user_id=user_id or "anonymous",
                title=f"Handwritten Math {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        # Add user interaction
        interaction_id = conversation_repo["add_interaction"](
            conversation_id=conversation_id,
            user_input={"type": "image", "filename": file.filename},
            system_response={"status": "processing"}
        )
        
        # Start handwriting recognition workflow
        workflow_id = orchestration_manager["start_workflow"](
            workflow_type="handwriting_recognition",
            initial_data={
                "image_path": file_path,
                "conversation_id": conversation_id,
                "interaction_id": interaction_id,
                "user_id": user_id
            }
        )
        
        # Return response with workflow ID for tracking
        return MathResponse(
            workflow_id=workflow_id,
            status="processing",
            message="Your handwritten expression is being processed"
        )
    
    except Exception as e:
        logger.error(f"Error processing handwritten math: {e}")
        raise HTTPException(status_code=500, detail=str(e))

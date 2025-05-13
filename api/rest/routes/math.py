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
from core.agent.llm_agent import CoreLLMAgent

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

class MathQueryResult(BaseModel):
    """Response model for math query results."""
    workflow_id: str
    query: str
    response: str
    status: str = "completed"
    duration_ms: float = 0

# Store for workflow results
math_workflow_results = {}

# Create LLM agent instance
llm_agent = CoreLLMAgent()

# Dependency for getting system components
async def get_orchestration_manager():
    """Get the orchestration manager instance."""
    # Our simple implementation that uses the CoreLLMAgent directly
    def start_workflow(workflow_type: str, initial_data: Dict[str, Any], **kwargs):
        workflow_id = str(uuid.uuid4())
        
        # Process the request in a background task
        if workflow_type == "math_problem_solving":
            # Start a background task to process this
            query = initial_data.get("query", "")
            BackgroundTasks().add_task(process_math_query, workflow_id, query)
            
        return workflow_id
    
    return {"start_workflow": start_workflow}

async def get_conversation_repository():
    """Get the conversation repository instance."""
    # Simple implementation that tracks conversations
    return {
        "create_conversation": lambda *args, **kwargs: str(uuid.uuid4()),
        "add_interaction": lambda *args, **kwargs: str(uuid.uuid4())
    }

# Background task for processing math queries
def process_math_query(workflow_id: str, query: str):
    """Process a math query in the background using the LLM agent."""
    try:
        logger.info(f"Processing math query: {query} (workflow_id: {workflow_id})")
        
        # Generate response from LLM
        start_time = datetime.now()
        result = llm_agent.generate_response(query)
        end_time = datetime.now()
        
        # Calculate duration
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Store the result
        if result.get("success", False):
            math_workflow_results[workflow_id] = {
                "workflow_id": workflow_id,
                "query": query,
                "response": result.get("response", ""),
                "status": "completed",
                "duration_ms": duration_ms
            }
        else:
            math_workflow_results[workflow_id] = {
                "workflow_id": workflow_id,
                "query": query,
                "response": f"Error: {result.get('error', 'Unknown error')}",
                "status": "failed",
                "duration_ms": duration_ms
            }
        
        logger.info(f"Math query processed in {duration_ms:.2f}ms (workflow_id: {workflow_id})")
        
    except Exception as e:
        logger.error(f"Error processing math query: {e}")
        math_workflow_results[workflow_id] = {
            "workflow_id": workflow_id,
            "query": query,
            "response": f"Error: {str(e)}",
            "status": "failed",
            "duration_ms": 0
    }

# Routes
@router.post("/query", response_model=MathResponse)
async def math_query(
    request: MathQueryRequest,
    background_tasks: BackgroundTasks,
    orchestration_manager: Any = Depends(get_orchestration_manager),
    conversation_repo: Any = Depends(get_conversation_repository)
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
        
        # Start math problem solving workflow
        workflow_id = str(uuid.uuid4())
        
        # Process in background task
        background_tasks.add_task(
            process_math_query,
            workflow_id,
            request.query
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

@router.get("/query/result/{workflow_id}", response_model=MathQueryResult)
async def get_math_query_result(workflow_id: str):
    """
    Get the result of a math query by workflow ID.
    
    Args:
        workflow_id: ID of the workflow to retrieve
        
    Returns:
        Result of the math query
    """
    # Check if the result exists
    if workflow_id not in math_workflow_results:
        # Check if it's still processing
        return MathQueryResult(
            workflow_id=workflow_id,
            query="",
            response="",
            status="processing",
            duration_ms=0
        )
    
    # Return the result
    result = math_workflow_results[workflow_id]
    return MathQueryResult(
        workflow_id=result["workflow_id"],
        query=result["query"],
        response=result["response"],
        status=result["status"],
        duration_ms=result["duration_ms"]
    )

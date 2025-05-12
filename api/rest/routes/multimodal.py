"""
API routes for multimodal input processing.

This module provides REST API endpoints for handling multimodal
input processing, clarification, and feedback.
"""
import os
import base64
import logging
import tempfile
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body, BackgroundTasks, File, UploadFile, Form, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from multimodal.unified_pipeline.input_processor import InputProcessor
from multimodal.unified_pipeline.content_router import ContentRouter
from multimodal.context.context_manager import ContextManager
from multimodal.interaction.ambiguity_handler import AmbiguityHandler
from multimodal.interaction.feedback_processor import FeedbackProcessor
from orchestration.manager.orchestration_manager import OrchestrationManager

# Initialize router
router = APIRouter(
    prefix="/multimodal",
    tags=["multimodal"],
    responses={404: {"description": "Not found"}},
)

# Initialize components
input_processor = InputProcessor()
content_router = ContentRouter()
context_manager = ContextManager()
ambiguity_handler = AmbiguityHandler()
feedback_processor = FeedbackProcessor()

# Get orchestration manager instance
from orchestration.manager.orchestration_manager import get_orchestration_manager
orchestration_manager = get_orchestration_manager()

logger = logging.getLogger(__name__)


# Define models
class TextInput(BaseModel):
    """Text input model."""
    content: str = Field(..., description="Text content")
    content_type: Optional[str] = Field("text/plain", description="Content type")


class ImageInput(BaseModel):
    """Image input model."""
    content: str = Field(..., description="Base64 encoded image content")
    encoding: str = Field("base64", description="Encoding type")
    mime_type: str = Field("image/png", description="MIME type")


class InputPart(BaseModel):
    """Input part model for multipart input."""
    input_type: str = Field(..., description="Input type (text, image)")
    content: Dict[str, Any] = Field(..., description="Content based on input type")


class MultipartInput(BaseModel):
    """Multipart input model."""
    parts: Dict[str, InputPart] = Field(..., description="Input parts by key")


class InputRequest(BaseModel):
    """Input request model."""
    input_type: str = Field(..., description="Input type (text, image, multipart)")
    content: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Content based on input type")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    context_id: Optional[str] = Field(None, description="Context ID")


class ClarificationAction(BaseModel):
    """Clarification action model."""
    action: str = Field(..., description="Action (confirm, edit, edit_symbol, retry, etc.)")
    latex: Optional[str] = Field(None, description="Corrected LaTeX if action is edit")
    symbol_id: Optional[int] = Field(None, description="Symbol ID if action is edit_symbol")
    new_text: Optional[str] = Field(None, description="New text for symbol if action is edit_symbol")


class ClarificationRequest(BaseModel):
    """Clarification request model."""
    session_id: str = Field(..., description="Session ID")
    input_id: str = Field(..., description="Input ID")
    clarification: ClarificationAction = Field(..., description="Clarification action")


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    type: str = Field(..., description="Feedback type (correction, rating, preference, error_report)")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    input_id: Optional[str] = Field(None, description="Input ID")
    original: Optional[Dict[str, Any]] = Field(None, description="Original content")
    correction: Optional[Dict[str, Any]] = Field(None, description="Corrected content")
    rating: Optional[int] = Field(None, description="Rating (1-5)")
    comment: Optional[str] = Field(None, description="Comment")


@router.post("/input")
async def process_input(input_request: InputRequest):
    """
    Process multimodal input.
    
    Handles text, image, or multipart input and starts appropriate processing workflow.
    """
    start_time = datetime.now()
    
    # Validate content
    if not input_request.content:
        raise HTTPException(status_code=400, detail="Empty content is not allowed")
    
    try:
        input_type = input_request.input_type
        content = input_request.content
        
        # Prepare input based on type
        if input_type == "text":
            if not isinstance(content, str):
                raise HTTPException(status_code=400, detail="Text input requires string content")
                
            processed_input = input_processor.process_input(content, "text/plain")
            
        elif input_type == "image":
            if not isinstance(content, dict):
                raise HTTPException(status_code=400, detail="Image input requires dictionary content")
                
            # Decode base64 image
            encoding = content.get("encoding", "base64")
            mime_type = content.get("mime_type", "image/png")
            image_data = content.get("data")
            
            if encoding != "base64":
                raise HTTPException(status_code=400, detail=f"Unsupported encoding: {encoding}")
                
            try:
                decoded_image = base64.b64decode(image_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")
                
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
                temp.write(decoded_image)
                temp_path = temp.name
            
            try:
                processed_input = input_processor.process_input(temp_path, mime_type)
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        elif input_type == "multipart":
            if not isinstance(content, dict):
                raise HTTPException(status_code=400, detail="Multipart input requires dictionary content")
                
            # Process each part
            parts = {}
            temp_files = []
            
            try:
                for key, part in content.get("parts", {}).items():
                    part_type = part.get("input_type")
                    part_content = part.get("content")
                    
                    if part_type == "text":
                        parts[key] = input_processor.process_input(part_content, "text/plain")
                    elif part_type == "image":
                        # Decode base64 image
                        encoding = part.get("encoding", "base64")
                        mime_type = part.get("mime_type", "image/png")
                        image_data = part.get("data")
                        
                        if encoding != "base64":
                            raise HTTPException(status_code=400, detail=f"Unsupported encoding: {encoding}")
                            
                        decoded_image = base64.b64decode(image_data)
                        
                        # Save to temporary file for processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
                            temp.write(decoded_image)
                            temp_path = temp.name
                            temp_files.append(temp_path)
                        
                        parts[key] = input_processor.process_input(temp_path, mime_type)
                    else:
                        raise HTTPException(status_code=400, detail=f"Unsupported part type: {part_type}")
                
                # Combine parts into multipart input
                processed_input = {
                    "success": True,
                    "input_type": "multipart",
                    "parts": parts
                }
            finally:
                # Clean up temp files
                for temp_path in temp_files:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported input type: {input_type}")
        
        # Check for ambiguities
        ambiguities = ambiguity_handler.detect_ambiguities(processed_input)
        
        if ambiguities.get("has_ambiguities", False):
            # Generate clarification request
            clarification_request = ambiguity_handler.generate_clarification_request(ambiguities)
            
            # Add clarification request to response
            processed_input["needs_clarification"] = True
            processed_input["clarification_request"] = clarification_request
        else:
            processed_input["needs_clarification"] = False
        
        # Route content to appropriate agents if no clarification needed
        if not processed_input.get("needs_clarification", False):
            routing_result = content_router.route_content(processed_input)
            processed_input["routing"] = routing_result
            
            # Add to context if context_id provided
            if input_request.context_id:
                entity_data = {
                    "type": input_type,
                    "source": "api",
                    "processed_data": processed_input
                }
                
                if input_type == "text":
                    entity_data["text"] = content
                elif input_type == "image":
                    entity_data["mime_type"] = content.get("mime_type", "image/png")
                
                entity_id = context_manager.add_entity_to_context(
                    input_request.context_id,
                    entity_data,
                    input_type
                )
                
                if entity_id:
                    processed_input["entity_id"] = entity_id
            
            # Start workflow in orchestration manager
            if input_request.conversation_id:
                workflow_id = await orchestration_manager.start_workflow(
                    workflow_type="multimodal_processing",
                    initial_data={
                        "processed_input": processed_input,
                        "conversation_id": input_request.conversation_id,
                        "context_id": input_request.context_id
                    }
                )
                
                processed_input["workflow_id"] = workflow_id
        
        # Add processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        processed_input["processing_time_ms"] = round(processing_time, 2)
        
        return processed_input
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")


@router.post("/clarify")
async def handle_clarification(clarification_request: ClarificationRequest):
    """
    Handle user clarification for ambiguous input.
    
    Processes user clarification and continues with the workflow.
    """
    start_time = datetime.now()
    
    try:
        session_id = clarification_request.session_id
        input_id = clarification_request.input_id
        clarification = clarification_request.clarification
        
        # Retrieve the original input from storage
        # In a real implementation, this would use a proper storage system
        # For this example, we'll use the orchestration manager to retrieve it
        
        original_input = orchestration_manager.get_processing_state(session_id, input_id)
        
        if not original_input:
            raise HTTPException(status_code=404, detail="Original input not found")
        
        # Process the clarification
        updated_input = ambiguity_handler.process_clarification(original_input, clarification.dict())
        
        # Update the processing state
        orchestration_manager.update_processing_state(session_id, input_id, updated_input)
        
        # Route content to appropriate agents
        routing_result = content_router.route_content(updated_input)
        updated_input["routing"] = routing_result
        
        # Resume workflow
        workflow_id = orchestration_manager.resume_workflow(
            session_id=session_id,
            input_id=input_id,
            updated_data=updated_input
        )
        
        updated_input["workflow_id"] = workflow_id
        
        # Add processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        updated_input["processing_time_ms"] = round(processing_time, 2)
        
        return updated_input
        
    except Exception as e:
        logger.error(f"Error handling clarification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error handling clarification: {str(e)}")


@router.post("/feedback")
async def process_user_feedback(feedback_request: FeedbackRequest):
    """
    Process user feedback.
    
    Handles various types of feedback including corrections, ratings, and error reports.
    """
    try:
        # Process feedback
        result = feedback_processor.process_feedback(feedback_request.dict())
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")


@router.post("/upload")
async def upload_file(file: UploadFile, 
                     conversation_id: Optional[str] = Form(None),
                     context_id: Optional[str] = Form(None)):
    """
    Upload a file for multimodal processing.
    
    Alternative to the /input endpoint for direct file uploads.
    """
    start_time = datetime.now()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Get file type
        mime_type = file.content_type
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp.write(file_content)
            temp_path = temp.name
        
        try:
            # Process the file
            if mime_type.startswith("image/"):
                processed_input = input_processor.process_input(temp_path, mime_type)
            elif mime_type.startswith("text/"):
                # For text files, read the content
                with open(temp_path, "r") as f:
                    text_content = f.read()
                processed_input = input_processor.process_input(text_content, mime_type)
            elif mime_type == "application/pdf":
                processed_input = input_processor.process_input(temp_path, mime_type)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")
                
            # The rest of the processing is the same as in the /input endpoint
            # Check for ambiguities
            ambiguities = ambiguity_handler.detect_ambiguities(processed_input)
            
            if ambiguities.get("has_ambiguities", False):
                # Generate clarification request
                clarification_request = ambiguity_handler.generate_clarification_request(ambiguities)
                
                # Add clarification request to response
                processed_input["needs_clarification"] = True
                processed_input["clarification_request"] = clarification_request
            else:
                processed_input["needs_clarification"] = False
            
            # Route content to appropriate agents if no clarification needed
            if not processed_input.get("needs_clarification", False):
                routing_result = content_router.route_content(processed_input)
                processed_input["routing"] = routing_result
                
                # Add to context if context_id provided
                if context_id:
                    entity_data = {
                        "type": "file",
                        "source": "api",
                        "file_name": file.filename,
                        "mime_type": mime_type,
                        "processed_data": processed_input
                    }
                    
                    input_type = "image" if mime_type.startswith("image/") else "text"
                    
                    entity_id = context_manager.add_entity_to_context(
                        context_id,
                        entity_data,
                        input_type
                    )
                    
                    if entity_id:
                        processed_input["entity_id"] = entity_id
                
                # Start workflow in orchestration manager
                if conversation_id:
                    workflow_id = orchestration_manager.start_workflow(
                        workflow_type="multimodal_processing",
                        initial_data={
                            "processed_input": processed_input,
                            "conversation_id": conversation_id,
                            "context_id": context_id
                        }
                    )
                    
                    processed_input["workflow_id"] = workflow_id
            
            # Add original filename and processing time
            processed_input["original_filename"] = file.filename
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            processed_input["processing_time_ms"] = round(processing_time, 2)
            
            return processed_input
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file upload: {str(e)}")


@router.get("/context/{context_id}")
async def get_context(context_id: str):
    """
    Get the current context.
    
    Retrieves the current state of a specific context.
    """
    try:
        context = context_manager.get_context(context_id)
        
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
        
        return context.to_dict()
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

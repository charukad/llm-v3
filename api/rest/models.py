"""API request/response models."""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class InputRequest(BaseModel):
    """Request model for input processing."""
    content: str = Field(..., description="The content to process")
    input_type: str = Field(..., description="Type of input (text, image, multipart)")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation")
    context_id: Optional[str] = Field(None, description="ID of the context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata") 
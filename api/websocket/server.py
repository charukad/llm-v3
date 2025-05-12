"""
WebSocket server for real-time notifications.

This module sets up WebSocket endpoints for real-time notifications
related to multimodal processing.
"""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import uuid

from .multimodal_handler import handle_websocket_connection

logger = logging.getLogger(__name__)

# Create router
websocket_router = APIRouter()

@websocket_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time notifications.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await handle_websocket_connection(websocket, client_id)

@websocket_router.websocket("/ws")
async def websocket_endpoint_auto_id(websocket: WebSocket):
    """
    WebSocket endpoint that automatically generates a client ID.
    
    Args:
        websocket: WebSocket connection
    """
    client_id = str(uuid.uuid4())
    await handle_websocket_connection(websocket, client_id)

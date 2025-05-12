"""
WebSocket handler for multimodal processing notifications.

This module handles WebSocket connections for providing real-time
updates on multimodal processing.
"""
import logging
import json
import asyncio
from typing import Dict, Any, List, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class MultimodalNotificationManager:
    """Manager for multimodal processing notifications via WebSocket."""
    
    def __init__(self):
        """Initialize the notification manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_subscribers: Dict[str, Set[str]] = {}
        logger.info("Initialized multimodal notification manager")
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Connect a client WebSocket.
        
        Args:
            websocket: The client WebSocket
            client_id: Client identifier
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """
        Disconnect a client.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Remove from all subscriptions
            for session_id, subscribers in self.session_subscribers.items():
                if client_id in subscribers:
                    subscribers.remove(client_id)
            
            logger.info(f"Client disconnected: {client_id}")
    
    def subscribe_to_session(self, client_id: str, session_id: str):
        """
        Subscribe a client to a session.
        
        Args:
            client_id: Client identifier
            session_id: Session identifier
        """
        if session_id not in self.session_subscribers:
            self.session_subscribers[session_id] = set()
            
        self.session_subscribers[session_id].add(client_id)
        logger.info(f"Client {client_id} subscribed to session {session_id}")
    
    def unsubscribe_from_session(self, client_id: str, session_id: str):
        """
        Unsubscribe a client from a session.
        
        Args:
            client_id: Client identifier
            session_id: Session identifier
        """
        if session_id in self.session_subscribers and client_id in self.session_subscribers[session_id]:
            self.session_subscribers[session_id].remove(client_id)
            logger.info(f"Client {client_id} unsubscribed from session {session_id}")
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """
        Broadcast a message to all clients subscribed to a session.
        
        Args:
            session_id: Session identifier
            message: Message to broadcast
        """
        if session_id not in self.session_subscribers:
            return
            
        disconnected_clients = []
        
        for client_id in self.session_subscribers[session_id]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to client {client_id}: {str(e)}")
                    disconnected_clients.append(client_id)
            else:
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

# Create a global instance
notification_manager = MultimodalNotificationManager()

async def handle_websocket_connection(websocket: WebSocket, client_id: str):
    """
    Handle a WebSocket connection.
    
    Args:
        websocket: The client WebSocket
        client_id: Client identifier
    """
    await notification_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process the message
            message_type = data.get("type", "")
            
            if message_type == "subscribe":
                # Subscribe to session updates
                session_id = data.get("session_id", "")
                if session_id:
                    notification_manager.subscribe_to_session(client_id, session_id)
                    
                    # Send confirmation
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "session_id": session_id
                    })
                    
            elif message_type == "unsubscribe":
                # Unsubscribe from session updates
                session_id = data.get("session_id", "")
                if session_id:
                    notification_manager.unsubscribe_from_session(client_id, session_id)
                    
                    # Send confirmation
                    await websocket.send_json({
                        "type": "unsubscription_confirmed",
                        "session_id": session_id
                    })
                    
            else:
                # Unknown message type
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        notification_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        notification_manager.disconnect(client_id)

async def send_processing_update(session_id: str, update_type: str, data: Dict[str, Any]):
    """
    Send a processing update to clients.
    
    Args:
        session_id: Session identifier
        update_type: Type of update
        data: Update data
    """
    message = {
        "type": "processing_update",
        "update_type": update_type,
        "session_id": session_id,
        "timestamp": str(datetime.datetime.now()),
        "data": data
    }
    
    await notification_manager.broadcast_to_session(session_id, message)

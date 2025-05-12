from typing import Dict, Any, Optional
import json
import time
import uuid
from datetime import datetime

class VisualizationMessageHandler:
    """
    Handles messages for the Visualization Agent according to the Multi-agent Communication Protocol (MCP).
    """
    
    def __init__(self, agent):
        """
        Initialize the message handler.
        
        Args:
            agent: The visualization agent instance
        """
        self.agent = agent
        self.message_types = {
            "visualization_request": self._handle_visualization_request,
            "capabilities_query": self._handle_capabilities_query
        }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message according to the MCP.
        
        Args:
            message: The incoming message
            
        Returns:
            Response message
        """
        try:
            # Extract message information
            header = message.get("header", {})
            message_id = header.get("message_id", str(uuid.uuid4()))
            sender = header.get("sender", "unknown")
            message_type = header.get("message_type", "")
            
            # Check if message type is supported
            if message_type not in self.message_types:
                return self._create_error_response(
                    message_id, 
                    sender,
                    f"Unsupported message type: {message_type}",
                    supported_types=list(self.message_types.keys())
                )
            
            # Process the message with the appropriate handler
            handler = self.message_types[message_type]
            response = handler(message)
            
            return response
            
        except Exception as e:
            return self._create_error_response(
                message.get("header", {}).get("message_id", str(uuid.uuid4())),
                message.get("header", {}).get("sender", "unknown"),
                f"Error processing message: {str(e)}"
            )
    
    def _handle_visualization_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle visualization request messages."""
        # Extract message information
        header = message.get("header", {})
        message_id = header.get("message_id", str(uuid.uuid4()))
        sender = header.get("sender", "unknown")
        
        # Process with agent
        result = self.agent.process_message(message)
        
        # Create response
        response = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "correlation_id": message_id,
                "sender": "visualization_agent",
                "recipient": sender,
                "timestamp": datetime.now().isoformat(),
                "message_type": "visualization_response"
            },
            "body": result
        }
        
        return response
    
    def _handle_capabilities_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle capabilities query messages."""
        # Extract message information
        header = message.get("header", {})
        message_id = header.get("message_id", str(uuid.uuid4()))
        sender = header.get("sender", "unknown")
        
        # Get agent capabilities
        capabilities = self.agent.get_capabilities()
        
        # Create response
        response = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "correlation_id": message_id,
                "sender": "visualization_agent",
                "recipient": sender,
                "timestamp": datetime.now().isoformat(),
                "message_type": "capabilities_response"
            },
            "body": {
                "capabilities": capabilities
            }
        }
        
        return response
    
    def _create_error_response(
        self, 
        correlation_id: str, 
        recipient: str, 
        error_message: str,
        **additional_info
    ) -> Dict[str, Any]:
        """Create an error response message."""
        response = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "sender": "visualization_agent",
                "recipient": recipient,
                "timestamp": datetime.now().isoformat(),
                "message_type": "error_response"
            },
            "body": {
                "success": False,
                "error": error_message
            }
        }
        
        # Add any additional information
        response["body"].update(additional_info)
        
        return response

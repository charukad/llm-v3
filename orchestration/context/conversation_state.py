"""
Conversation state management for the Mathematical Multimodal LLM System.

This module handles tracking the state of conversations, including messages,
timing information, and token usage.
"""

import time
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


class Message:
    """Represents a single message in a conversation."""
    
    def __init__(self, 
                message_id: str,
                role: str,
                content: str,
                timestamp: float,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a message.
        
        Args:
            message_id: Unique identifier for the message
            role: Role of the message sender (user, system, assistant)
            content: Content of the message
            timestamp: Timestamp when the message was created
            metadata: Optional metadata about the message
        """
        self.message_id = message_id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata or {}
    
    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Args:
            include_metadata: Whether to include message metadata
            
        Returns:
            Dictionary representation of the message
        """
        message_dict = {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }
        
        if include_metadata and self.metadata:
            message_dict["metadata"] = self.metadata
        
        return message_dict
    
    def estimate_token_count(self) -> int:
        """
        Estimate the number of tokens in the message.
        
        This is a simple approximation. For more accurate counts,
        you would use a proper tokenizer.
        
        Returns:
            Estimated token count
        """
        # Simple approximation: 1 token ≈ 4 characters for English text
        return len(self.content) // 4 + 5  # Add 5 tokens for message overhead
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """
        Update message metadata.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self.metadata.update(updates)


class ConversationState:
    """
    Manages the state of a conversation.
    
    Tracks messages, token usage, and other state information for a
    conversation.
    """
    
    def __init__(self, 
                conversation_id: str,
                user_id: str,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize conversation state.
        
        Args:
            conversation_id: Unique identifier for the conversation
            user_id: ID of the user in the conversation
            metadata: Optional metadata about the conversation
        """
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.messages = []  # List of Message objects
        self.message_map = {}  # message_id -> Message object
        self.created_at = time.time()
        self.updated_at = self.created_at
    
    def add_user_message(self, 
                        content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a user message to the conversation.
        
        Args:
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        timestamp = time.time()
        
        message = Message(message_id, "user", content, timestamp, metadata)
        self.messages.append(message)
        self.message_map[message_id] = message
        
        self.updated_at = timestamp
        
        return message_id
    
    def add_system_message(self, 
                          content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a system message to the conversation.
        
        Args:
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        timestamp = time.time()
        
        message = Message(message_id, "system", content, timestamp, metadata)
        self.messages.append(message)
        self.message_map[message_id] = message
        
        self.updated_at = timestamp
        
        return message_id
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.
        
        Args:
            message_id: ID of the message to get
            
        Returns:
            Message object if found, None otherwise
        """
        return self.message_map.get(message_id)
    
    def update_message_metadata(self, 
                               message_id: str, 
                               updates: Dict[str, Any]) -> bool:
        """
        Update a message's metadata.
        
        Args:
            message_id: ID of the message to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        message = self.get_message(message_id)
        if not message:
            return False
        
        message.update_metadata(updates)
        return True
    
    def get_context_text(self, 
                        include_metadata: bool = False,
                        max_tokens: Optional[int] = None) -> str:
        """
        Get the conversation context as text.
        
        Args:
            include_metadata: Whether to include message metadata
            max_tokens: Maximum number of tokens to include (from most recent)
            
        Returns:
            Text representation of the conversation context
        """
        # Start with most recent messages if max_tokens is specified
        messages_to_include = self.messages
        if max_tokens is not None:
            messages_to_include = self._get_messages_within_token_budget(max_tokens)
        
        # Generate text representation
        lines = []
        for message in messages_to_include:
            role_prefix = f"{message.role.upper()}: "
            content = message.content
            
            if include_metadata and message.metadata:
                metadata_str = json.dumps(message.metadata)
                lines.append(f"{role_prefix}{content} [Metadata: {metadata_str}]")
            else:
                lines.append(f"{role_prefix}{content}")
        
        return "\n\n".join(lines)
    
    def get_context_dict(self, 
                        include_metadata: bool = True,
                        max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the conversation context as a dictionary.
        
        Args:
            include_metadata: Whether to include message metadata
            max_tokens: Maximum number of tokens to include (from most recent)
            
        Returns:
            Dictionary representation of the conversation context
        """
        # Start with most recent messages if max_tokens is specified
        messages_to_include = self.messages
        if max_tokens is not None:
            messages_to_include = self._get_messages_within_token_budget(max_tokens)
        
        # Generate dictionary representation
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [
                message.to_dict(include_metadata) for message in messages_to_include
            ],
            "metadata": self.metadata
        }
    
    def estimate_token_count(self) -> int:
        """
        Estimate the total token count for the conversation.
        
        Returns:
            Estimated token count
        """
        # Sum token counts for all messages
        message_tokens = sum(message.estimate_token_count() for message in self.messages)
        
        # Add overhead for conversation structure
        overhead_tokens = 10  # Approximate overhead
        
        return message_tokens + overhead_tokens
    
    def get_message_count(self) -> int:
        """
        Get the number of messages in the conversation.
        
        Returns:
            Message count
        """
        return len(self.messages)
    
    def remove_message(self, message_id: str) -> bool:
        """
        Remove a message from the conversation.
        
        Args:
            message_id: ID of the message to remove
            
        Returns:
            True if successful, False otherwise
        """
        message = self.message_map.get(message_id)
        if not message:
            return False
        
        self.messages.remove(message)
        del self.message_map[message_id]
        
        return True
    
    def remove_messages(self, message_ids: List[str]) -> int:
        """
        Remove multiple messages from the conversation.
        
        Args:
            message_ids: List of message IDs to remove
            
        Returns:
            Number of messages successfully removed
        """
        removed_count = 0
        
        for message_id in message_ids:
            if self.remove_message(message_id):
                removed_count += 1
        
        return removed_count
    
    def _get_messages_within_token_budget(self, max_tokens: int) -> List[Message]:
        """
        Get as many recent messages as possible within a token budget.
        
        Args:
            max_tokens: Maximum number of tokens
            
        Returns:
            List of messages within the token budget, most recent first
        """
        if max_tokens <= 0:
            return []
        
        # Start with the most recent message
        messages_included = []
        tokens_used = 0
        
        # Iterate through messages in reverse order (newest first)
        for message in reversed(self.messages):
            message_tokens = message.estimate_token_count()
            
            # Check if adding this message would exceed the budget
            if tokens_used + message_tokens <= max_tokens:
                messages_included.insert(0, message)  # Insert at beginning to maintain order
                tokens_used += message_tokens
            else:
                # Can't include more messages
                break
        
        return messages_included

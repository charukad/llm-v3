"""
Context manager for the Mathematical Multimodal LLM System.

This module provides the central context management functionality, coordinating
conversation state tracking, entity management, reference resolution, and
context pruning strategies.
"""

import time
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from orchestration.monitoring.logger import get_logger
from orchestration.context.conversation_state import ConversationState
from orchestration.context.entity_tracker import EntityTracker
from orchestration.context.pruning_strategy import PruningStrategy, TokenBudgetStrategy

logger = get_logger(__name__)


class ContextManager:
    """
    Central manager for conversation context in the Mathematical Multimodal LLM System.
    
    The ContextManager maintains the state of conversations, tracks mathematical
    entities, resolves references, and implements pruning strategies to ensure
    the context remains within manageable limits.
    """
    
    def __init__(self, 
                 max_context_tokens: int = 4096,
                 entity_tracking_enabled: bool = True,
                 pruning_strategy: Optional[PruningStrategy] = None):
        """
        Initialize the context manager.
        
        Args:
            max_context_tokens: Maximum number of tokens in the context window
            entity_tracking_enabled: Whether to enable mathematical entity tracking
            pruning_strategy: Strategy for pruning context when it exceeds limits
        """
        self.max_context_tokens = max_context_tokens
        self.entity_tracking_enabled = entity_tracking_enabled
        self.pruning_strategy = pruning_strategy or TokenBudgetStrategy(max_context_tokens)
        
        # Initialize components
        self.conversation_states = {}  # conversation_id -> ConversationState
        self.entity_trackers = {}      # conversation_id -> EntityTracker
        
        logger.info(f"Initialized context manager with max_context_tokens={max_context_tokens}")
    
    def create_conversation(self, 
                           user_id: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            user_id: ID of the user initiating the conversation
            metadata: Optional metadata about the conversation
            
        Returns:
            New conversation ID
        """
        # Generate a new conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Initialize conversation state
        state = ConversationState(conversation_id, user_id, metadata)
        self.conversation_states[conversation_id] = state
        
        # Initialize entity tracker if enabled
        if self.entity_tracking_enabled:
            self.entity_trackers[conversation_id] = EntityTracker()
        
        logger.info(f"Created new conversation {conversation_id} for user {user_id}")
        
        return conversation_id
    
    def add_user_message(self, 
                        conversation_id: str, 
                        message: str,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a user message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            message: User message text
            metadata: Optional metadata about the message
            
        Returns:
            Updated context information
        """
        # Get conversation state
        state = self._get_conversation_state(conversation_id)
        
        # Add message to state
        message_id = state.add_user_message(message, metadata)
        
        # Process message for mathematical entities if enabled
        if self.entity_tracking_enabled:
            entity_tracker = self._get_entity_tracker(conversation_id)
            extracted_entities = entity_tracker.extract_entities(message)
            
            # Update message metadata with extracted entities
            if extracted_entities:
                state.update_message_metadata(message_id, {"entities": extracted_entities})
                logger.debug(f"Extracted entities from message {message_id}: {extracted_entities}")
        
        # Apply pruning if necessary
        pruned = self._apply_pruning_if_needed(conversation_id)
        
        # Return updated context information
        return {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "context_tokens": state.estimate_token_count(),
            "pruned": pruned
        }
    
    def add_system_message(self, 
                          conversation_id: str, 
                          message: str,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a system message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            message: System message text
            metadata: Optional metadata about the message
            
        Returns:
            Updated context information
        """
        # Get conversation state
        state = self._get_conversation_state(conversation_id)
        
        # Add message to state
        message_id = state.add_system_message(message, metadata)
        
        # Process message for mathematical entities if enabled
        if self.entity_tracking_enabled:
            entity_tracker = self._get_entity_tracker(conversation_id)
            extracted_entities = entity_tracker.extract_entities(message)
            
            # Track generated entities
            if extracted_entities:
                state.update_message_metadata(message_id, {"entities": extracted_entities})
                logger.debug(f"Extracted entities from system message {message_id}: {extracted_entities}")
        
        # Apply pruning if necessary
        pruned = self._apply_pruning_if_needed(conversation_id)
        
        # Return updated context information
        return {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "context_tokens": state.estimate_token_count(),
            "pruned": pruned
        }
    
    def get_conversation_context(self, 
                               conversation_id: str, 
                               format: str = "text",
                               include_metadata: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Get the current context for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            format: Format to return the context in ('text', 'dict', 'json')
            include_metadata: Whether to include message metadata
            
        Returns:
            Conversation context in the specified format
        """
        # Get conversation state
        state = self._get_conversation_state(conversation_id)
        
        # Get context in the requested format
        if format == "text":
            return state.get_context_text(include_metadata=include_metadata)
        elif format == "dict":
            return state.get_context_dict(include_metadata=include_metadata)
        elif format == "json":
            context_dict = state.get_context_dict(include_metadata=include_metadata)
            return json.dumps(context_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def resolve_entity_references(self, 
                                conversation_id: str, 
                                query: str) -> Dict[str, Any]:
        """
        Resolve entity references in a query.
        
        Args:
            conversation_id: ID of the conversation
            query: Query text potentially containing references
            
        Returns:
            Dictionary with resolved query and referenced entities
        """
        # Skip if entity tracking is disabled
        if not self.entity_tracking_enabled:
            return {"resolved_query": query, "referenced_entities": {}}
        
        # Get entity tracker
        entity_tracker = self._get_entity_tracker(conversation_id)
        
        # Resolve references
        resolved_query, referenced_entities = entity_tracker.resolve_references(query)
        
        logger.debug(f"Resolved references in query: {query} -> {resolved_query}")
        
        return {
            "resolved_query": resolved_query,
            "referenced_entities": referenced_entities
        }
    
    def get_relevant_entities(self, 
                             conversation_id: str, 
                             query: str,
                             max_entities: int = 5) -> List[Dict[str, Any]]:
        """
        Get entities relevant to a query.
        
        Args:
            conversation_id: ID of the conversation
            query: Query text
            max_entities: Maximum number of entities to return
            
        Returns:
            List of relevant entities with metadata
        """
        # Skip if entity tracking is disabled
        if not self.entity_tracking_enabled:
            return []
        
        # Get entity tracker
        entity_tracker = self._get_entity_tracker(conversation_id)
        
        # Get relevant entities
        return entity_tracker.get_relevant_entities(query, max_entities)
    
    def update_entity(self,
                     conversation_id: str,
                     entity_id: str,
                     updates: Dict[str, Any]) -> bool:
        """
        Update an entity's metadata.
        
        Args:
            conversation_id: ID of the conversation
            entity_id: ID of the entity to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if entity tracking is disabled
        if not self.entity_tracking_enabled:
            return False
        
        # Get entity tracker
        entity_tracker = self._get_entity_tracker(conversation_id)
        
        # Update the entity
        success = entity_tracker.update_entity(entity_id, updates)
        
        if success:
            logger.debug(f"Updated entity {entity_id} in conversation {conversation_id}")
        
        return success
    
    def get_context_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Summary of the context state
        """
        # Get conversation state
        state = self._get_conversation_state(conversation_id)
        
        # Get entity tracker if available
        entity_summary = {}
        if self.entity_tracking_enabled:
            entity_tracker = self._get_entity_tracker(conversation_id)
            entity_summary = entity_tracker.get_summary()
        
        # Build summary
        return {
            "conversation_id": conversation_id,
            "user_id": state.user_id,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "message_count": state.get_message_count(),
            "token_count": state.estimate_token_count(),
            "token_limit": self.max_context_tokens,
            "entity_tracking_enabled": self.entity_tracking_enabled,
            "entity_summary": entity_summary,
            "pruning_strategy": self.pruning_strategy.get_name(),
            "metadata": state.metadata
        }
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a conversation's history.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if conversation_id not in self.conversation_states:
            logger.warning(f"Attempted to clear non-existent conversation: {conversation_id}")
            return False
        
        # Get conversation state
        state = self.conversation_states[conversation_id]
        
        # Preserve metadata and other essential information
        user_id = state.user_id
        metadata = state.metadata
        
        # Create a new state with the same essential information
        self.conversation_states[conversation_id] = ConversationState(
            conversation_id, user_id, metadata)
        
        # Reset entity tracker if needed
        if self.entity_tracking_enabled and conversation_id in self.entity_trackers:
            self.entity_trackers[conversation_id] = EntityTracker()
        
        logger.info(f"Cleared conversation {conversation_id}")
        
        return True
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all associated data.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if conversation_id not in self.conversation_states:
            logger.warning(f"Attempted to delete non-existent conversation: {conversation_id}")
            return False
        
        # Remove the conversation state
        del self.conversation_states[conversation_id]
        
        # Remove entity tracker if it exists
        if conversation_id in self.entity_trackers:
            del self.entity_trackers[conversation_id]
        
        logger.info(f"Deleted conversation {conversation_id}")
        
        return True
    
    def _get_conversation_state(self, conversation_id: str) -> ConversationState:
        """
        Get the conversation state for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            ConversationState object
            
        Raises:
            ValueError: If the conversation doesn't exist
        """
        if conversation_id not in self.conversation_states:
            raise ValueError(f"Conversation does not exist: {conversation_id}")
        
        return self.conversation_states[conversation_id]
    
    def _get_entity_tracker(self, conversation_id: str) -> EntityTracker:
        """
        Get the entity tracker for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            EntityTracker object
            
        Raises:
            ValueError: If entity tracking is disabled or tracker doesn't exist
        """
        if not self.entity_tracking_enabled:
            raise ValueError("Entity tracking is disabled")
        
        if conversation_id not in self.entity_trackers:
            raise ValueError(f"Entity tracker does not exist for conversation: {conversation_id}")
        
        return self.entity_trackers[conversation_id]
    
    def _apply_pruning_if_needed(self, conversation_id: str) -> bool:
        """
        Apply pruning strategy if the context exceeds limits.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if pruning was applied, False otherwise
        """
        # Get conversation state
        state = self._get_conversation_state(conversation_id)
        
        # Check if pruning is needed
        current_tokens = state.estimate_token_count()
        
        if current_tokens > self.max_context_tokens:
            logger.info(f"Context for conversation {conversation_id} exceeds limit "
                      f"({current_tokens} > {self.max_context_tokens}), applying pruning")
            
            # Apply pruning strategy
            self.pruning_strategy.apply(state)
            
            # Log pruning results
            new_tokens = state.estimate_token_count()
            logger.info(f"Pruned conversation {conversation_id} context from "
                      f"{current_tokens} to {new_tokens} tokens")
            
            return True
        
        return False


# Singleton instance
_context_manager = None

def get_context_manager() -> ContextManager:
    """
    Get the singleton instance of the context manager.
    
    Returns:
        The context manager instance
    """
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager

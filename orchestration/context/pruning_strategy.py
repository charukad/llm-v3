"""
Context pruning strategies for the Mathematical Multimodal LLM System.

This module provides strategies for pruning conversation context when it
exceeds the maximum allowed size, ensuring efficient use of the context window.
"""

import abc
from typing import Dict, List, Any, Optional, Set, Tuple

from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


class PruningStrategy(abc.ABC):
    """Abstract base class for pruning strategies."""
    
    @abc.abstractmethod
    def apply(self, conversation_state) -> None:
        """
        Apply the pruning strategy to a conversation state.
        
        Args:
            conversation_state: ConversationState object to prune
        """
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the pruning strategy.
        
        Returns:
            Strategy name
        """
        pass


class TokenBudgetStrategy(PruningStrategy):
    """
    Pruning strategy based on token budget.
    
    Removes oldest messages first until the context is within budget.
    """
    
    def __init__(self, 
                max_tokens: int, 
                target_ratio: float = 0.8,
                preserve_system_messages: bool = True,
                preserve_last_n_turns: int = 2):
        """
        Initialize the token budget strategy.
        
        Args:
            max_tokens: Maximum token budget
            target_ratio: Target token usage ratio after pruning (0.0-1.0)
            preserve_system_messages: Whether to preserve system messages
            preserve_last_n_turns: Number of most recent turns to preserve
        """
        self.max_tokens = max_tokens
        self.target_tokens = int(max_tokens * target_ratio)
        self.preserve_system_messages = preserve_system_messages
        self.preserve_last_n_turns = preserve_last_n_turns
    
    def apply(self, conversation_state) -> None:
        """
        Apply the pruning strategy to a conversation state.
        
        Args:
            conversation_state: ConversationState object to prune
        """
        # Check current token count
        current_tokens = conversation_state.estimate_token_count()
        
        if current_tokens <= self.target_tokens:
            # No pruning needed
            return
        
        # Get all messages
        messages = conversation_state.messages.copy()
        if not messages:
            return
        
        # Determine how many messages to keep
        current_tokens = conversation_state.estimate_token_count()
        target_tokens = self.target_tokens
        
        # Always preserve the last N turns (1 turn = user message + system message)
        preserved_indices = set()
        if self.preserve_last_n_turns > 0:
            turns_preserved = 0
            user_found = False
            
            # Start from the end (most recent) and work backwards
            for i in range(len(messages) - 1, -1, -1):
                message = messages[i]
                
                if message.role == "user" and not user_found:
                    # Found a user message
                    preserved_indices.add(i)
                    user_found = True
                elif message.role == "system" and user_found:
                    # Found the system response to a user message
                    preserved_indices.add(i)
                    user_found = False
                    turns_preserved += 1
                    
                    if turns_preserved >= self.preserve_last_n_turns:
                        break
        
        # Identify messages that can be removed
        removable_messages = []
        for i, message in enumerate(messages):
            # Skip preserved messages
            if i in preserved_indices:
                continue
                
            # Skip system messages if configured to preserve them
            if self.preserve_system_messages and message.role == "system":
                continue
                
            removable_messages.append((i, message))
        
        # Sort removable messages by timestamp (oldest first)
        removable_messages.sort(key=lambda x: x[1].timestamp)
        
        # Remove messages until we're under the target token count
        removed_count = 0
        for _, message in removable_messages:
            message_tokens = message.estimate_token_count()
            message_id = message.message_id
            
            # Remove the message
            if conversation_state.remove_message(message_id):
                current_tokens -= message_tokens
                removed_count += 1
                
                if current_tokens <= target_tokens:
                    break
        
        logger.info(f"TokenBudgetStrategy removed {removed_count} messages, "
                  f"reducing token count from {current_tokens + message_tokens * removed_count} "
                  f"to {current_tokens}")
    
    def get_name(self) -> str:
        """
        Get the name of the pruning strategy.
        
        Returns:
            Strategy name
        """
        return "TokenBudgetStrategy"


class RelevancePruningStrategy(PruningStrategy):
    """
    Pruning strategy based on message relevance.
    
    Removes messages based on relevance to the current conversation topic.
    """
    
    def __init__(self, 
                max_tokens: int,
                target_ratio: float = 0.8,
                preserve_last_n_turns: int = 2):
        """
        Initialize the relevance pruning strategy.
        
        Args:
            max_tokens: Maximum token budget
            target_ratio: Target token usage ratio after pruning (0.0-1.0)
            preserve_last_n_turns: Number of most recent turns to preserve
        """
        self.max_tokens = max_tokens
        self.target_tokens = int(max_tokens * target_ratio)
        self.preserve_last_n_turns = preserve_last_n_turns
    
    def apply(self, conversation_state) -> None:
        """
        Apply the pruning strategy to a conversation state.
        
        Args:
            conversation_state: ConversationState object to prune
        """
        # Check current token count
        current_tokens = conversation_state.estimate_token_count()
        
        if current_tokens <= self.target_tokens:
            # No pruning needed
            return
        
        # Get all messages
        messages = conversation_state.messages.copy()
        if not messages:
            return
        
        # Always preserve the last N turns
        preserved_indices = set()
        if self.preserve_last_n_turns > 0:
            turns_preserved = 0
            user_found = False
            
            # Start from the end (most recent) and work backwards
            for i in range(len(messages) - 1, -1, -1):
                message = messages[i]
                
                if message.role == "user" and not user_found:
                    # Found a user message
                    preserved_indices.add(i)
                    user_found = True
                elif message.role == "system" and user_found:
                    # Found the system response to a user message
                    preserved_indices.add(i)
                    user_found = False
                    turns_preserved += 1
                    
                    if turns_preserved >= self.preserve_last_n_turns:
                        break
        
        # Calculate relevance scores for each message
        scores = []
        for i, message in enumerate(messages):
            # Skip preserved messages
            if i in preserved_indices:
                continue
                
            score = self._calculate_relevance_score(message, messages[-1])
            scores.append((i, message, score))
        
        # Sort by relevance score (lowest first)
        scores.sort(key=lambda x: x[2])
        
        # Remove messages until we're under the target token count
        removed_count = 0
        for _, message, score in scores:
            message_tokens = message.estimate_token_count()
            message_id = message.message_id
            
            # Remove the message
            if conversation_state.remove_message(message_id):
                current_tokens -= message_tokens
                removed_count += 1
                
                if current_tokens <= self.target_tokens:
                    break
        
        logger.info(f"RelevancePruningStrategy removed {removed_count} messages, "
                  f"reducing token count from {current_tokens + message_tokens * removed_count} "
                  f"to {current_tokens}")
    
    def _calculate_relevance_score(self, message, current_message) -> float:
        """
        Calculate relevance score for a message.
        
        Args:
            message: Message to score
            current_message: Current message for relevance comparison
            
        Returns:
            Relevance score (higher is more relevant)
        """
        # Simple relevance calculation based on text similarity
        # In a real implementation, you might use embeddings or more sophisticated methods
        
        # Extract words from both messages
        message_words = set(message.content.lower().split())
        current_words = set(current_message.content.lower().split())
        
        # Calculate overlap
        if not message_words or not current_words:
            return 0.0
        
        overlap = len(message_words.intersection(current_words))
        score = overlap / max(len(message_words), len(current_words))
        
        return score
    
    def get_name(self) -> str:
        """
        Get the name of the pruning strategy.
        
        Returns:
            Strategy name
        """
        return "RelevancePruningStrategy"


class SummaryPruningStrategy(PruningStrategy):
    """
    Pruning strategy that summarizes older parts of the conversation.
    
    Replaces older message groups with summaries to save tokens.
    """
    
    def __init__(self, 
                max_tokens: int,
                target_ratio: float = 0.8,
                llm_summarizer = None):
        """
        Initialize the summary pruning strategy.
        
        Args:
            max_tokens: Maximum token budget
            target_ratio: Target token usage ratio after pruning (0.0-1.0)
            llm_summarizer: LLM-based summarizer function
        """
        self.max_tokens = max_tokens
        self.target_tokens = int(max_tokens * target_ratio)
        self.llm_summarizer = llm_summarizer
    
    def apply(self, conversation_state) -> None:
        """
        Apply the pruning strategy to a conversation state.
        
        Args:
            conversation_state: ConversationState object to prune
        """
        # Check if we have a summarizer function
        if not self.llm_summarizer:
            logger.warning("SummaryPruningStrategy requires a summarizer function")
            # Fall back to token budget strategy
            fallback = TokenBudgetStrategy(self.max_tokens)
            fallback.apply(conversation_state)
            return
        
        # Check current token count
        current_tokens = conversation_state.estimate_token_count()
        
        if current_tokens <= self.target_tokens:
            # No pruning needed
            return
        
        # Get all messages
        messages = conversation_state.messages.copy()
        if len(messages) < 4:  # Need at least 2 turns to summarize
            return
        
        # Group messages into conversation turns (user message + system response)
        turns = []
        i = 0
        while i < len(messages):
            # Look for a user message
            if i < len(messages) and messages[i].role == "user":
                user_msg = messages[i]
                i += 1
                
                # Look for the corresponding system message
                if i < len(messages) and messages[i].role == "system":
                    system_msg = messages[i]
                    i += 1
                    turns.append((user_msg, system_msg))
                else:
                    # Unpaired user message
                    turns.append((user_msg, None))
            else:
                # Skip other message types
                i += 1
        
        # If we have a very short conversation, don't summarize yet
        if len(turns) < 2:
            return
        
        # Determine how many turns to summarize (oldest first)
        # Start by summarizing half of the conversation except the most recent turn
        num_turns_to_summarize = max(1, len(turns) // 2)
        turns_to_summarize = turns[:num_turns_to_summarize]
        
        # Prepare text for summarization
        summary_text = ""
        for user_msg, system_msg in turns_to_summarize:
            summary_text += f"User: {user_msg.content}\n"
            if system_msg:
                summary_text += f"System: {system_msg.content}\n\n"
        
        # Generate summary
        summary = self.llm_summarizer(summary_text)
        
        # Replace summarized messages with a summary
        message_ids_to_remove = []
        for user_msg, system_msg in turns_to_summarize:
            message_ids_to_remove.append(user_msg.message_id)
            if system_msg:
                message_ids_to_remove.append(system_msg.message_id)
        
        # Remove messages that will be summarized
        conversation_state.remove_messages(message_ids_to_remove)
        
        # Add summary as a system message at the beginning
        conversation_state.add_system_message(
            f"[Summary of previous conversation: {summary}]",
            metadata={"is_summary": True, "summarized_messages": len(message_ids_to_remove)}
        )
        
        logger.info(f"SummaryPruningStrategy summarized {len(message_ids_to_remove)} messages "
                  f"into a summary message")
    
    def get_name(self) -> str:
        """
        Get the name of the pruning strategy.
        
        Returns:
            Strategy name
        """
        return "SummaryPruningStrategy"

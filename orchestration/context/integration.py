"""
Integration of context management with other components of the system.

This module provides functions for integrating the context management system
with the orchestration manager, workflow engine, and other components.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from orchestration.monitoring.logger import get_logger
from orchestration.context.context_manager import get_context_manager
from orchestration.context.entity_tracker import EntityTracker
from orchestration.context.pruning_strategy import (
    TokenBudgetStrategy, 
    RelevancePruningStrategy,
    SummaryPruningStrategy
)

logger = get_logger(__name__)


def integrate_with_orchestration_manager(orchestration_manager):
    """
    Integrate context management with the orchestration manager.
    
    Args:
        orchestration_manager: Orchestration manager to integrate with
    """
    logger.info("Integrating context management with orchestration manager")
    
    # Get the context manager
    context_manager = get_context_manager()
    
    # Register message processing hooks if available
    if hasattr(orchestration_manager, 'register_message_hook'):
        orchestration_manager.register_message_hook(
            'pre_process', _pre_process_message_hook)
        orchestration_manager.register_message_hook(
            'post_process', _post_process_message_hook)
        logger.info("Registered message processing hooks")
    
    # Register workflow hooks if available
    if hasattr(orchestration_manager, 'register_workflow_hook'):
        orchestration_manager.register_workflow_hook(
            'pre_execution', _pre_workflow_execution_hook)
        orchestration_manager.register_workflow_hook(
            'post_execution', _post_workflow_execution_hook)
        logger.info("Registered workflow hooks")


def _pre_process_message_hook(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook called before message processing to add context.
    
    Args:
        message: Message to process
        
    Returns:
        Modified message
    """
    # Skip if message already has conversation_id
    if "conversation_id" not in message.get("header", {}):
        return message
    
    # Get conversation_id from header
    conversation_id = message["header"]["conversation_id"]
    
    # Get context manager
    context_manager = get_context_manager()
    
    try:
        # Resolve references in the message
        if "content" in message.get("body", {}):
            content = message["body"]["content"]
            
            # Resolve references
            resolved = context_manager.resolve_entity_references(conversation_id, content)
            
            # Update message with resolved content
            message["body"]["content"] = resolved["resolved_query"]
            
            # Add referenced entities to message metadata
            if "metadata" not in message["body"]:
                message["body"]["metadata"] = {}
            
            message["body"]["metadata"]["referenced_entities"] = resolved["referenced_entities"]
            
            logger.debug(f"Resolved references in message for conversation {conversation_id}")
        
        # Add relevant context to the message
        if "query" in message.get("body", {}):
            query = message["body"]["query"]
            
            # Get relevant entities
            relevant_entities = context_manager.get_relevant_entities(
                conversation_id, query, max_entities=3)
            
            # Add relevant entities to message
            if "context" not in message["body"]:
                message["body"]["context"] = {}
            
            message["body"]["context"]["relevant_entities"] = relevant_entities
            
            logger.debug(f"Added {len(relevant_entities)} relevant entities to message context")
        
    except Exception as e:
        logger.warning(f"Error applying context in pre-process hook: {str(e)}")
    
    return message


def _post_process_message_hook(message: Dict[str, Any], 
                              response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook called after message processing to update context.
    
    Args:
        message: Original message
        response: Response to the message
        
    Returns:
        Modified response
    """
    # Skip if message has no conversation_id
    if "conversation_id" not in message.get("header", {}):
        return response
    
    # Get conversation_id from header
    conversation_id = message["header"]["conversation_id"]
    
    # Get context manager
    context_manager = get_context_manager()
    
    try:
        # Add the response to the conversation context
        if "content" in response.get("body", {}):
            content = response["body"]["content"]
            
            # Add as system message
            metadata = response.get("body", {}).get("metadata", {})
            
            context_manager.add_system_message(
                conversation_id, content, metadata)
            
            logger.debug(f"Added response to conversation {conversation_id} context")
        
        # Extract and track entities from the response
        if "content" in response.get("body", {}):
            content = response["body"]["content"]
            
            # Extract entities from response (use entity_tracker directly for more control)
            entity_tracker = EntityTracker()
            extracted_entities = entity_tracker.extract_entities(content)
            
            # Add extracted entities to response metadata
            if "metadata" not in response["body"]:
                response["body"]["metadata"] = {}
            
            response["body"]["metadata"]["extracted_entities"] = extracted_entities
            
            logger.debug(f"Extracted {len(extracted_entities)} entities from response")
    
    except Exception as e:
        logger.warning(f"Error updating context in post-process hook: {str(e)}")
    
    return response


def _pre_workflow_execution_hook(workflow_execution) -> None:
    """
    Hook called before workflow execution to add context.
    
    Args:
        workflow_execution: Workflow execution
    """
    # Skip if workflow has no conversation_id
    if not hasattr(workflow_execution, 'conversation_id'):
        return
    
    conversation_id = workflow_execution.conversation_id
    
    # Get context manager
    context_manager = get_context_manager()
    
    try:
        # Get conversation context
        context_dict = context_manager.get_conversation_context(
            conversation_id, format="dict")
        
        # Add context to workflow metadata
        if not hasattr(workflow_execution, 'metadata'):
            workflow_execution.metadata = {}
        
        workflow_execution.metadata["conversation_context"] = context_dict
        
        logger.debug(f"Added conversation context to workflow {workflow_execution.workflow_id}")
    
    except Exception as e:
        logger.warning(f"Error adding context in pre-workflow hook: {str(e)}")


def _post_workflow_execution_hook(workflow_execution) -> None:
    """
    Hook called after workflow execution to update context.
    
    Args:
        workflow_execution: Workflow execution
    """
    # Skip if workflow has no conversation_id
    if not hasattr(workflow_execution, 'conversation_id'):
        return
    
    conversation_id = workflow_execution.conversation_id
    
    # Skip if workflow didn't produce a result
    if not hasattr(workflow_execution, 'result') or not workflow_execution.result:
        return
    
    # Get context manager
    context_manager = get_context_manager()
    
    try:
        # Extract entities from workflow result
        result = workflow_execution.result
        
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            
            # Add result as system message
            metadata = result.get("metadata", {})
            metadata["workflow_id"] = workflow_execution.workflow_id
            
            context_manager.add_system_message(
                conversation_id, content, metadata)
            
            logger.debug(f"Added workflow result to conversation {conversation_id} context")
        
    except Exception as e:
        logger.warning(f"Error updating context in post-workflow hook: {str(e)}")


def initialize_context_management(max_context_tokens: int = 4096) -> None:
    """
    Initialize all context management components.
    
    Args:
        max_context_tokens: Maximum context token limit
    """
    # Get the context manager
    context_manager = get_context_manager()
    
    # Configure the context manager
    context_manager.max_context_tokens = max_context_tokens
    context_manager.entity_tracking_enabled = True
    
    # Set pruning strategy
    context_manager.pruning_strategy = TokenBudgetStrategy(
        max_tokens=max_context_tokens,
        target_ratio=0.8,
        preserve_system_messages=True,
        preserve_last_n_turns=2
    )
    
    logger.info(f"Initialized context management with max_tokens={max_context_tokens}")


# Initialize on module load
initialize_context_management()

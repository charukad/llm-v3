"""
Cross-modal context manager for maintaining context across different input modalities.

This module handles context tracking and reference resolution between
text and visual content.
"""
import uuid
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class CrossModalContext:
    """
    Cross-modal context object representing a single interaction context.
    Contains references to entities across different modalities.
    """
    
    def __init__(self, context_id: Optional[str] = None):
        """
        Initialize a cross-modal context.
        
        Args:
            context_id: Optional context ID, generated if not provided
        """
        self.context_id = context_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.entities = {}  # Dictionary of entities by ID
        self.references = {}  # Cross-references between entities
        self.modalities = {}  # Entities by modality
        self.previous_contexts = []  # List of previous context IDs in conversation
    
    def add_entity(self, entity_id: str, entity_data: Dict[str, Any], 
                  modality: str) -> str:
        """
        Add an entity to the context.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_data: Entity data dictionary
            modality: The modality of the entity (text, image, etc.)
            
        Returns:
            The entity ID
        """
        # Store the entity
        self.entities[entity_id] = entity_data
        
        # Add to modality index
        if modality not in self.modalities:
            self.modalities[modality] = []
        
        self.modalities[modality].append(entity_id)
        
        # Update timestamp
        self.updated_at = datetime.now()
        
        return entity_id
    
    def add_reference(self, source_id: str, target_id: str, 
                     ref_type: str) -> Dict[str, Any]:
        """
        Add a reference between entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            ref_type: Type of reference
            
        Returns:
            The created reference object
        """
        if source_id not in self.entities or target_id not in self.entities:
            raise ValueError(f"Both source and target entities must exist")
        
        reference = {
            "source_id": source_id,
            "target_id": target_id,
            "type": ref_type,
            "created_at": datetime.now()
        }
        
        reference_id = f"{source_id}_{target_id}_{ref_type}"
        self.references[reference_id] = reference
        
        # Update timestamp
        self.updated_at = datetime.now()
        
        return reference
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data or None if not found
        """
        return self.entities.get(entity_id)
    
    def get_entities_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific modality.
        
        Args:
            modality: The modality to filter by
            
        Returns:
            List of entities
        """
        entity_ids = self.modalities.get(modality, [])
        return [self.entities[entity_id] for entity_id in entity_ids
                if entity_id in self.entities]
    
    def find_references(self, entity_id: str, ref_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find all references involving an entity.
        
        Args:
            entity_id: The entity ID
            ref_type: Optional reference type to filter by
            
        Returns:
            List of reference objects
        """
        references = []
        
        for ref in self.references.values():
            if (ref["source_id"] == entity_id or ref["target_id"] == entity_id) and \
               (ref_type is None or ref["type"] == ref_type):
                references.append(ref)
        
        return references
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the context to a dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "context_id": self.context_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "entities": self.entities,
            "references": self.references,
            "modalities": self.modalities,
            "previous_contexts": self.previous_contexts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrossModalContext':
        """
        Create a context object from a dictionary.
        
        Args:
            data: Dictionary representation of a context
            
        Returns:
            A new CrossModalContext object
        """
        context = cls(context_id=data.get("context_id"))
        
        # Convert datetime strings to datetime objects
        context.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        context.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        
        # Load data
        context.entities = data.get("entities", {})
        context.references = data.get("references", {})
        context.modalities = data.get("modalities", {})
        context.previous_contexts = data.get("previous_contexts", [])
        
        return context


class ContextManager:
    """
    Manager for cross-modal contexts.
    Handles creation, retrieval, and persistence of contexts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.active_contexts = {}  # Context ID -> CrossModalContext
        logger.info("Initialized cross-modal context manager")
    
    def create_context(self, conversation_id: Optional[str] = None,
                      previous_context_id: Optional[str] = None) -> CrossModalContext:
        """
        Create a new context.
        
        Args:
            conversation_id: Optional conversation ID for linking
            previous_context_id: Optional previous context ID
            
        Returns:
            New CrossModalContext object
        """
        context = CrossModalContext()
        
        # Link to conversation if provided
        if conversation_id:
            context.conversation_id = conversation_id
        
        # Link to previous context if provided
        if previous_context_id and previous_context_id in self.active_contexts:
            context.previous_contexts.append(previous_context_id)
            
            # Copy relevant entities and references from previous context
            prev_context = self.active_contexts[previous_context_id]
            
            # Copy certain entity types that should persist
            for entity_id, entity_data in prev_context.entities.items():
                entity_type = entity_data.get("type", "")
                if entity_type in ["variable", "expression", "theorem", "concept"]:
                    context.entities[entity_id] = entity_data
                    
                    # Also copy modality entries
                    for modality, ids in prev_context.modalities.items():
                        if entity_id in ids:
                            if modality not in context.modalities:
                                context.modalities[modality] = []
                            context.modalities[modality].append(entity_id)
        
        # Store in active contexts
        self.active_contexts[context.context_id] = context
        
        logger.info(f"Created new context with ID: {context.context_id}")
        return context
    
    def get_context(self, context_id: str) -> Optional[CrossModalContext]:
        """
        Get a context by ID.
        
        Args:
            context_id: The context ID
            
        Returns:
            CrossModalContext or None if not found
        """
        if context_id in self.active_contexts:
            return self.active_contexts[context_id]
        
        # In a real implementation, this would attempt to load from storage
        logger.warning(f"Context not found: {context_id}")
        return None
    
    def update_context(self, context: CrossModalContext) -> None:
        """
        Update a context in storage.
        
        Args:
            context: The context to update
        """
        context.updated_at = datetime.now()
        self.active_contexts[context.context_id] = context
        
        # In a real implementation, this would persist to storage
        logger.info(f"Updated context: {context.context_id}")
    
    def add_entity_to_context(self, context_id: str, entity_data: Dict[str, Any],
                             modality: str, entity_id: Optional[str] = None) -> Optional[str]:
        """
        Add an entity to a context.
        
        Args:
            context_id: The context ID
            entity_data: Entity data dictionary
            modality: The modality of the entity
            entity_id: Optional entity ID, generated if not provided
            
        Returns:
            Entity ID or None if context not found
        """
        context = self.get_context(context_id)
        if not context:
            return None
        
        # Generate entity ID if not provided
        entity_id = entity_id or str(uuid.uuid4())
        
        # Add to context
        context.add_entity(entity_id, entity_data, modality)
        
        # Update context in storage
        self.update_context(context)
        
        return entity_id
    
    def add_reference_to_context(self, context_id: str, source_id: str,
                               target_id: str, ref_type: str) -> Optional[Dict[str, Any]]:
        """
        Add a reference between entities in a context.
        
        Args:
            context_id: The context ID
            source_id: Source entity ID
            target_id: Target entity ID
            ref_type: Type of reference
            
        Returns:
            Reference object or None if context not found
        """
        context = self.get_context(context_id)
        if not context:
            return None
        
        try:
            # Add reference
            reference = context.add_reference(source_id, target_id, ref_type)
            
            # Update context in storage
            self.update_context(context)
            
            return reference
        
        except ValueError as e:
            logger.error(f"Error adding reference: {str(e)}")
            return None
    
    def resolve_references(self, context_id: str, query: str) -> Dict[str, Any]:
        """
        Resolve references in a query using context.
        
        Args:
            context_id: The context ID
            query: The query containing references
            
        Returns:
            Dictionary with resolved references
        """
        context = self.get_context(context_id)
        if not context:
            return {"success": False, "error": "Context not found"}
        
        # In a real implementation, this would use NLP to identify and resolve references
        # For this example, we'll just return a simple response
        
        # Get all text entities that might be referenced
        text_entities = context.get_entities_by_modality("text")
        image_entities = context.get_entities_by_modality("image")
        
        return {
            "success": True,
            "original_query": query,
            "resolved_query": query,  # In a real implementation, this would be updated
            "referenced_entities": {
                "text": [entity.get("id") for entity in text_entities],
                "image": [entity.get("id") for entity in image_entities]
            }
        }

# Singleton instance for the context manager
_context_manager_instance = None

def get_context_manager() -> ContextManager:
    """Get or create the context manager singleton instance."""
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = ContextManager()
    return _context_manager_instance

# Add method to get conversation context
async def get_conversation_context(self, conversation_id: str, 
                                context_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get context for a conversation.
    
    Args:
        conversation_id: The conversation ID
        context_id: Optional specific context ID
        
    Returns:
        Dictionary with context data
    """
    # If a specific context is requested, get that
    if context_id:
        context = self.get_context(context_id)
        if context:
            return {
                "conversation_id": conversation_id,
                "context_id": context_id,
                "context_data": context.to_dict()
            }
    
    # Otherwise, create a new context for this conversation
    context = self.create_context(conversation_id=conversation_id)
    
    # In a real implementation, this would fetch conversation history
    # from a database or other storage
    
    return {
        "conversation_id": conversation_id,
        "context_id": context.context_id,
        "context_data": context.to_dict(),
        "conversation_history": []
    }

# Add the method to the ContextManager class
ContextManager.get_conversation_context = get_conversation_context

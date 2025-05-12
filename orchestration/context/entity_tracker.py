"""
Entity tracking and reference resolution for Mathematical Multimodal LLM System.

This module provides tracking for mathematical entities (expressions, variables,
theorems, etc.) mentioned in conversations and resolves references to these
entities.
"""

import re
import uuid
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import json

from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


class MathematicalEntity:
    """Represents a mathematical entity that can be referenced in conversations."""
    
    def __init__(self, 
                entity_id: str,
                entity_type: str,
                value: str,
                display_form: Optional[str] = None,
                latex_form: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a mathematical entity.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity (expression, variable, theorem, etc.)
            value: Value or definition of the entity
            display_form: User-friendly display form (if different from value)
            latex_form: LaTeX representation (if available)
            metadata: Additional metadata about the entity
        """
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.value = value
        self.display_form = display_form or value
        self.latex_form = latex_form
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.last_referenced_at = self.created_at
        self.reference_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary.
        
        Returns:
            Dictionary representation of the entity
        """
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "value": self.value,
            "display_form": self.display_form,
            "latex_form": self.latex_form,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_referenced_at": self.last_referenced_at,
            "reference_count": self.reference_count
        }
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """
        Update entity metadata.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self.metadata.update(updates)
    
    def record_reference(self) -> None:
        """Record that this entity was referenced."""
        self.last_referenced_at = time.time()
        self.reference_count += 1


class EntityTracker:
    """
    Tracks mathematical entities and resolves references to them.
    
    This class is responsible for:
    1. Extracting mathematical entities from text
    2. Tracking these entities throughout a conversation
    3. Resolving references to entities in subsequent queries
    4. Finding entities relevant to a query
    """
    
    def __init__(self):
        """Initialize the entity tracker."""
        self.entities = {}  # entity_id -> MathematicalEntity
        self.entities_by_type = {}  # entity_type -> set of entity_ids
        self.named_entities = {}  # name -> entity_id
        
        # Reference patterns for different types of entities
        self.reference_patterns = {
            # Expression references
            "expression": [
                r"(that|the|this|previous|above|last) (expression|equation|formula)",
                r"expression (\d+)",
                r"equation (\d+)"
            ],
            # Variable references
            "variable": [
                r"(that|the|this|previous|above|last) variable",
                r"variable (\w+)"
            ],
            # Function references
            "function": [
                r"(that|the|this|previous|above|last) function",
                r"function (\w+)",
                r"the function (\w+)\(.*?\)"
            ],
            # Matrix references
            "matrix": [
                r"(that|the|this|previous|above|last) matrix",
                r"matrix (\w+)"
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        extracted_entities = []
        
        # Extract expressions enclosed in dollar signs (LaTeX math mode)
        latex_expressions = self._extract_latex_expressions(text)
        for latex in latex_expressions:
            entity_id = self._add_entity("expression", latex, latex_form=latex)
            extracted_entities.append({
                "entity_id": entity_id,
                "entity_type": "expression",
                "value": latex,
                "latex_form": latex
            })
        
        # Extract explicitly defined variables
        variables = self._extract_variable_definitions(text)
        for var_name, var_value in variables:
            # Check if the variable has a LaTeX form
            latex_form = None
            for latex in latex_expressions:
                if var_name in latex:
                    latex_form = latex
                    break
            
            entity_id = self._add_entity(
                "variable", 
                var_value, 
                display_form=var_name,
                latex_form=latex_form
            )
            
            # Add to named entities for easy reference
            self.named_entities[var_name] = entity_id
            
            extracted_entities.append({
                "entity_id": entity_id,
                "entity_type": "variable",
                "value": var_value,
                "display_form": var_name,
                "latex_form": latex_form
            })
        
        # Extract explicitly defined functions
        functions = self._extract_function_definitions(text)
        for func_name, func_def in functions:
            # Check if the function has a LaTeX form
            latex_form = None
            for latex in latex_expressions:
                if func_name in latex:
                    latex_form = latex
                    break
            
            entity_id = self._add_entity(
                "function", 
                func_def, 
                display_form=func_name,
                latex_form=latex_form
            )
            
            # Add to named entities for easy reference
            self.named_entities[func_name] = entity_id
            
            extracted_entities.append({
                "entity_id": entity_id,
                "entity_type": "function",
                "value": func_def,
                "display_form": func_name,
                "latex_form": latex_form
            })
        
        # More entity types could be added here
        
        return extracted_entities
    
    def resolve_references(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve references to entities in a query.
        
        Args:
            query: Query text potentially containing references
            
        Returns:
            Tuple of (resolved query, referenced entities dict)
        """
        resolved_query = query
        referenced_entities = {}
        
        # Check for references using patterns
        for entity_type, patterns in self.reference_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                
                for match in matches:
                    # Get relevant entities of this type
                    relevant_entities = self._get_entities_by_type(entity_type)
                    if not relevant_entities:
                        continue
                    
                    # Handle different reference types
                    if "previous" in match.group(0) or "last" in match.group(0) or "above" in match.group(0):
                        # Get the most recently created entity of this type
                        entity = self._get_most_recent_entity(entity_type)
                    elif match.group(0).startswith(entity_type) and len(match.groups()) > 0:
                        # Reference by name or number
                        ref_id = match.group(1)
                        entity = self._get_entity_by_reference(entity_type, ref_id)
                    else:
                        # Generic reference, get most recently referenced entity
                        entity = self._get_most_recently_referenced_entity(entity_type)
                    
                    if entity:
                        # Replace the reference in the query
                        replacement = entity.display_form
                        if entity.latex_form:
                            replacement = entity.latex_form
                        
                        resolved_query = resolved_query.replace(match.group(0), replacement)
                        
                        # Record the reference
                        entity.record_reference()
                        
                        # Add to referenced entities
                        referenced_entities[entity.entity_id] = entity.to_dict()
        
        # Also check for direct references to named entities
        for name, entity_id in self.named_entities.items():
            if name in query:
                entity = self.entities.get(entity_id)
                if entity:
                    # Record the reference
                    entity.record_reference()
                    
                    # Add to referenced entities
                    referenced_entities[entity.entity_id] = entity.to_dict()
        
        return resolved_query, referenced_entities
    
    def get_relevant_entities(self, 
                             query: str,
                             max_entities: int = 5) -> List[Dict[str, Any]]:
        """
        Get entities relevant to a query.
        
        Args:
            query: Query text
            max_entities: Maximum number of entities to return
            
        Returns:
            List of relevant entity dictionaries
        """
        if not self.entities:
            return []
        
        # Calculate relevance scores for each entity
        scored_entities = []
        for entity_id, entity in self.entities.items():
            score = self._calculate_relevance_score(entity, query)
            scored_entities.append((entity, score))
        
        # Sort by relevance score (descending)
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top entities
        top_entities = [entity.to_dict() for entity, _ in scored_entities[:max_entities]]
        return top_entities
    
    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an entity's metadata.
        
        Args:
            entity_id: ID of the entity to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        entity = self.entities.get(entity_id)
        if not entity:
            return False
        
        # Apply updates
        if "metadata" in updates:
            entity.update_metadata(updates["metadata"])
        
        if "value" in updates:
            entity.value = updates["value"]
        
        if "display_form" in updates:
            entity.display_form = updates["display_form"]
        
        if "latex_form" in updates:
            entity.latex_form = updates["latex_form"]
        
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: ID of the entity to get
            
        Returns:
            Entity dictionary if found, None otherwise
        """
        entity = self.entities.get(entity_id)
        if not entity:
            return None
        
        return entity.to_dict()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tracked entities.
        
        Returns:
            Summary dictionary
        """
        entity_counts = {}
        for entity_type in self.entities_by_type:
            entity_counts[entity_type] = len(self.entities_by_type[entity_type])
        
        return {
            "total_entities": len(self.entities),
            "entity_counts": entity_counts,
            "named_entities": len(self.named_entities)
        }
    
    def _add_entity(self, 
                   entity_type: str, 
                   value: str,
                   display_form: Optional[str] = None,
                   latex_form: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new entity to the tracker.
        
        Args:
            entity_type: Type of entity
            value: Value or definition of the entity
            display_form: User-friendly display form
            latex_form: LaTeX representation
            metadata: Additional metadata
            
        Returns:
            Entity ID
        """
        entity_id = str(uuid.uuid4())
        
        entity = MathematicalEntity(
            entity_id, 
            entity_type, 
            value, 
            display_form, 
            latex_form, 
            metadata
        )
        
        self.entities[entity_id] = entity
        
        # Add to type-based index
        if entity_type not in self.entities_by_type:
            self.entities_by_type[entity_type] = set()
        self.entities_by_type[entity_type].add(entity_id)
        
        return entity_id
    
    def _extract_latex_expressions(self, text: str) -> List[str]:
        """
        Extract LaTeX expressions from text.
        
        Looks for expressions enclosed in dollar signs or \begin{equation} blocks.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of LaTeX expressions
        """
        expressions = []
        
        # Find inline math expressions ($ ... $)
        inline_matches = re.finditer(r'\$(.*?)\$', text)
        for match in inline_matches:
            expressions.append(match.group(1))
        
        # Find display math expressions (\begin{equation} ... \end{equation})
        display_matches = re.finditer(
            r'\\begin\{equation\}(.*?)\\end\{equation\}', 
            text, 
            re.DOTALL
        )
        for match in display_matches:
            expressions.append(match.group(1))
        
        # Find display math expressions with dollar pairs ($$ ... $$)
        display_dollar_matches = re.finditer(r'\$\$(.*?)\$\$', text, re.DOTALL)
        for match in display_dollar_matches:
            expressions.append(match.group(1))
        
        return expressions
    
    def _extract_variable_definitions(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract variable definitions from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of (variable name, variable value) tuples
        """
        variables = []
        
        # Look for "let X be..." or "define X as..." patterns
        let_patterns = [
            r'let\s+(\w+)\s+be\s+(.*?)(?:\.|$)',
            r'define\s+(\w+)\s+as\s+(.*?)(?:\.|$)',
            r'set\s+(\w+)\s+=\s+(.*?)(?:\.|$)',
            r'(\w+)\s+is defined as\s+(.*?)(?:\.|$)'
        ]
        
        for pattern in let_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                var_name = match.group(1)
                var_value = match.group(2).strip()
                variables.append((var_name, var_value))
        
        # Look for assignment patterns like "X = ..."
        assignment_matches = re.finditer(r'(\w+)\s*=\s*(.*?)(?:,|\.|$|and)', text)
        for match in assignment_matches:
            var_name = match.group(1)
            var_value = match.group(2).strip()
            variables.append((var_name, var_value))
        
        return variables
    
    def _extract_function_definitions(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract function definitions from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of (function name, function definition) tuples
        """
        functions = []
        
        # Look for "let f(x) = ..." or "define f(x) as..." patterns
        let_patterns = [
            r'let\s+(\w+)\((\w+(?:,\s*\w+)*)\)\s*=\s*(.*?)(?:\.|$)',
            r'define\s+(\w+)\((\w+(?:,\s*\w+)*)\)\s*as\s*(.*?)(?:\.|$)',
            r'function\s+(\w+)\((\w+(?:,\s*\w+)*)\)\s*=\s*(.*?)(?:\.|$)'
        ]
        
        for pattern in let_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                func_name = match.group(1)
                func_args = match.group(2)
                func_body = match.group(3).strip()
                func_def = f"{func_name}({func_args}) = {func_body}"
                functions.append((func_name, func_def))
        
        # Look for direct function assignments
        assignment_matches = re.finditer(
            r'(\w+)\((\w+(?:,\s*\w+)*)\)\s*=\s*(.*?)(?:,|\.|$)', 
            text
        )
        for match in assignment_matches:
            func_name = match.group(1)
            func_args = match.group(2)
            func_body = match.group(3).strip()
            func_def = f"{func_name}({func_args}) = {func_body}"
            functions.append((func_name, func_def))
        
        return functions
    
    def _get_entities_by_type(self, entity_type: str) -> List[MathematicalEntity]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entity to get
            
        Returns:
            List of entities
        """
        if entity_type not in self.entities_by_type:
            return []
        
        return [self.entities[entity_id] for entity_id in self.entities_by_type[entity_type]]
    
    def _get_most_recent_entity(self, entity_type: str) -> Optional[MathematicalEntity]:
        """
        Get the most recently created entity of a specific type.
        
        Args:
            entity_type: Type of entity to get
            
        Returns:
            Most recent entity or None if no entities of that type
        """
        entities = self._get_entities_by_type(entity_type)
        if not entities:
            return None
        
        # Sort by creation time (most recent first)
        entities.sort(key=lambda e: e.created_at, reverse=True)
        return entities[0]
    
    def _get_most_recently_referenced_entity(self, 
                                           entity_type: str) -> Optional[MathematicalEntity]:
        """
        Get the most recently referenced entity of a specific type.
        
        Args:
            entity_type: Type of entity to get
            
        Returns:
            Most recently referenced entity or None if no entities of that type
        """
        entities = self._get_entities_by_type(entity_type)
        if not entities:
            return None
        
        # Sort by last referenced time (most recent first)
        entities.sort(key=lambda e: e.last_referenced_at, reverse=True)
        return entities[0]
    
    def _get_entity_by_reference(self, 
                               entity_type: str, 
                               reference: str) -> Optional[MathematicalEntity]:
        """
        Get an entity by a reference string.
        
        Args:
            entity_type: Type of entity to get
            reference: Reference string (name, number, etc.)
            
        Returns:
            Referenced entity or None if not found
        """
        # If reference is a number, interpret as index
        if reference.isdigit():
            # Get entities of this type sorted by creation time
            entities = self._get_entities_by_type(entity_type)
            if not entities:
                return None
            
            entities.sort(key=lambda e: e.created_at)
            
            # Use 1-indexed numbering (1 = oldest)
            index = int(reference) - 1
            if 0 <= index < len(entities):
                return entities[index]
            
            return None
        
        # Otherwise, check if it's a named entity
        entity_id = self.named_entities.get(reference)
        if entity_id:
            entity = self.entities.get(entity_id)
            if entity and entity.entity_type == entity_type:
                return entity
        
        # Try to match by display form
        for entity in self._get_entities_by_type(entity_type):
            if entity.display_form == reference:
                return entity
        
        return None
    
    def _calculate_relevance_score(self, entity: MathematicalEntity, query: str) -> float:
        """
        Calculate relevance score for an entity to a query.
        
        Args:
            entity: Entity to score
            query: Query text
            
        Returns:
            Relevance score (higher is more relevant)
        """
        score = 0.0
        
        # Check if entity appears in query
        if entity.display_form and entity.display_form in query:
            score += 1.0
        
        # Check if entity type is mentioned in query
        if entity.entity_type in query:
            score += 0.5
        
        # Check for value overlap
        query_tokens = set(query.lower().split())
        value_tokens = set(entity.value.lower().split())
        token_overlap = len(query_tokens.intersection(value_tokens))
        if token_overlap > 0:
            score += 0.1 * token_overlap
        
        # Consider recency and reference count
        time_factor = 1.0 / (1.0 + (time.time() - entity.last_referenced_at) / 3600)  # Decay over hours
        score += 0.3 * time_factor
        
        ref_count_factor = min(entity.reference_count / 5.0, 1.0)  # Cap at 5 references
        score += 0.2 * ref_count_factor
        
        return score

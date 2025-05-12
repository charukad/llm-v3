"""
Reference resolver for resolving references between different modalities.

This module handles entity matching, reference detection, and resolution
across text and visual inputs.
"""
import re
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

from .context_manager import CrossModalContext

logger = logging.getLogger(__name__)

class ReferenceResolver:
    """Resolves references between different modalities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reference resolver.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Common reference patterns
        self.reference_patterns = [
            # Explicit references to equations, figures, etc.
            r"(equation|figure|diagram|expression|formula)\s+(\d+)",
            
            # Positional references like "above", "below", etc.
            r"(above|below|previous|following|next)\s+(equation|figure|expression)",
            
            # Demonstrative references like "this equation", "that expression"
            r"(this|that|the)\s+(equation|figure|diagram|expression|formula)",
            
            # References with ordinals
            r"(first|second|third|last|next)\s+(equation|figure|expression)",
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.reference_patterns]
        
        logger.info("Initialized reference resolver")
    
    def resolve_references(self, context: CrossModalContext, text: str) -> Dict[str, Any]:
        """
        Resolve references in text using the provided context.
        
        Args:
            context: CrossModalContext containing entities and references
            text: Text containing references to resolve
            
        Returns:
            Dictionary with resolved references
        """
        # Extract potential references from text
        potential_refs = self._extract_references(text)
        
        # Find matching entities in context
        resolved_refs = {}
        
        for ref_text, ref_type, ref_id in potential_refs:
            # Find matching entities based on reference type
            matches = self._find_matching_entities(context, ref_type, ref_id)
            
            if matches:
                resolved_refs[ref_text] = {
                    "type": ref_type,
                    "id": ref_id,
                    "matches": matches
                }
        
        # Create resolved version of text
        resolved_text = self._create_resolved_text(text, resolved_refs)
        
        return {
            "original_text": text,
            "resolved_text": resolved_text,
            "resolved_references": resolved_refs
        }
    
    def find_cross_modal_links(self, context: CrossModalContext) -> List[Dict[str, Any]]:
        """
        Find and establish links between entities across different modalities.
        
        Args:
            context: CrossModalContext containing entities and references
            
        Returns:
            List of new cross-modal links established
        """
        new_links = []
        
        # Get entities by modality
        text_entities = context.get_entities_by_modality("text")
        image_entities = context.get_entities_by_modality("image")
        
        # For each image entity, find matching text entities
        for image_entity in image_entities:
            image_id = image_entity.get("id")
            if not image_id:
                continue
                
            image_latex = image_entity.get("recognized_latex", "")
            if not image_latex:
                continue
            
            # Find text entities that might match this image entity
            for text_entity in text_entities:
                text_id = text_entity.get("id")
                if not text_id:
                    continue
                    
                text_latex = text_entity.get("latex", "")
                if not text_latex:
                    continue
                
                # Check for similarity between latex expressions
                similarity = self._calculate_latex_similarity(image_latex, text_latex)
                
                if similarity > 0.7:  # Threshold for considering a match
                    # Create a cross-modal link
                    link = {
                        "source_id": text_id,
                        "target_id": image_id,
                        "type": "equivalent",
                        "confidence": similarity
                    }
                    
                    # Add to context
                    reference_id = f"{text_id}_{image_id}_equivalent"
                    if reference_id not in context.references:
                        context.references[reference_id] = link
                        new_links.append(link)
        
        return new_links
    
    def _extract_references(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract potential references from text.
        
        Args:
            text: Text to extract references from
            
        Returns:
            List of tuples (reference_text, reference_type, reference_id)
        """
        references = []
        
        # Apply all compiled patterns
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                reference_text = match.group(0)
                
                # Determine reference type and ID based on the pattern
                if "equation" in match.group(0).lower() or "expression" in match.group(0).lower() or "formula" in match.group(0).lower():
                    ref_type = "equation"
                elif "figure" in match.group(0).lower() or "diagram" in match.group(0).lower():
                    ref_type = "figure"
                else:
                    ref_type = "unknown"
                
                # Try to extract ID if present
                if len(match.groups()) > 1 and match.group(2).isdigit():
                    ref_id = match.group(2)
                else:
                    # For references like "above equation", use position
                    if "above" in match.group(0).lower() or "previous" in match.group(0).lower():
                        ref_id = "previous"
                    elif "below" in match.group(0).lower() or "next" in match.group(0).lower() or "following" in match.group(0).lower():
                        ref_id = "next"
                    elif "this" in match.group(0).lower() or "that" in match.group(0).lower():
                        ref_id = "current"
                    else:
                        ref_id = "unknown"
                
                references.append((reference_text, ref_type, ref_id))
        
        return references
    
    def _find_matching_entities(self, context: CrossModalContext, ref_type: str, 
                              ref_id: str) -> List[Dict[str, Any]]:
        """
        Find entities in context that match a reference type and ID.
        
        Args:
            context: CrossModalContext containing entities
            ref_type: Type of reference ("equation", "figure", etc.)
            ref_id: ID or positional reference
            
        Returns:
            List of matching entities
        """
        matches = []
        
        # Map reference types to entity types
        entity_types = {
            "equation": ["expression", "formula", "equation"],
            "figure": ["image", "diagram", "plot"],
            "unknown": []  # Match any type
        }
        
        target_types = entity_types.get(ref_type, [])
        
        # Get all entities
        all_entities = []
        for entity_id, entity_data in context.entities.items():
            entity_copy = entity_data.copy()
            entity_copy["id"] = entity_id
            all_entities.append(entity_copy)
        
        # Sort by creation time if available
        all_entities.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        
        # Filter by type if needed
        if target_types:
            filtered_entities = [e for e in all_entities 
                               if e.get("type", "") in target_types]
        else:
            filtered_entities = all_entities
        
        # Match by ID or position
        if ref_id.isdigit():
            # Explicit ID
            for entity in filtered_entities:
                if entity.get("number") == int(ref_id) or entity.get("id") == ref_id:
                    matches.append(entity)
        elif ref_id == "previous":
            # Previous entity of the target type
            if filtered_entities:
                matches.append(filtered_entities[0])
        elif ref_id == "next":
            # Next entity (not usually possible to resolve)
            pass
        elif ref_id == "current":
            # Most recent entity of target type
            if filtered_entities:
                matches.append(filtered_entities[0])
        else:
            # Unknown reference - return most recent entities
            matches = filtered_entities[:3]  # Top 3 most recent
        
        return matches
    
    def _create_resolved_text(self, text: str, resolved_refs: Dict[str, Any]) -> str:
        """
        Create a version of the text with references resolved.
        
        Args:
            text: Original text
            resolved_refs: Dictionary of resolved references
            
        Returns:
            Text with resolved references
        """
        # In a real implementation, this would replace references with more explicit versions
        # For simplicity, we'll just return the original text
        return text
    
    def _calculate_latex_similarity(self, latex1: str, latex2: str) -> float:
        """
        Calculate similarity between two LaTeX expressions.
        
        Args:
            latex1: First LaTeX expression
            latex2: Second LaTeX expression
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # In a real implementation, this would use a more sophisticated
        # algorithm for comparing LaTeX expressions
        
        # Normalize LaTeX by removing whitespace and common variations
        def normalize_latex(latex: str) -> str:
            # Remove whitespace
            latex = re.sub(r'\s+', '', latex)
            
            # Replace common equivalent notations
            latex = latex.replace('\\frac', '/')
            latex = latex.replace('{', '').replace('}', '')
            
            return latex
        
        norm1 = normalize_latex(latex1)
        norm2 = normalize_latex(latex2)
        
        # Basic string similarity
        if norm1 == norm2:
            return 1.0
        
        # Compute Jaccard similarity of character trigrams
        def get_trigrams(s: str) -> Set[str]:
            return {s[i:i+3] for i in range(len(s)-2)}
        
        trigrams1 = get_trigrams(norm1) if len(norm1) >= 3 else {norm1}
        trigrams2 = get_trigrams(norm2) if len(norm2) >= 3 else {norm2}
        
        intersection = trigrams1.intersection(trigrams2)
        union = trigrams1.union(trigrams2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)

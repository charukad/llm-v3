"""
Unit tests for cross-modal context management.

This module contains tests for the context manager and reference resolver
implemented for Sprint 12.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from multimodal.context.context_manager import ContextManager, CrossModalContext
from multimodal.context.reference_resolver import ReferenceResolver


class TestCrossModalContext(unittest.TestCase):
    """Tests for the cross-modal context."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = CrossModalContext()
    
    def test_add_entity(self):
        """Test adding an entity to context."""
        # Add text entity
        entity_id = self.context.add_entity(
            entity_id="entity1",
            entity_data={"type": "expression", "latex": "x^2"},
            modality="text"
        )
        
        # Verify entity was added
        self.assertEqual(entity_id, "entity1")
        self.assertIn("entity1", self.context.entities)
        self.assertIn("entity1", self.context.modalities.get("text", []))
    
    def test_add_reference(self):
        """Test adding a reference between entities."""
        # Add entities
        self.context.add_entity(
            entity_id="entity1",
            entity_data={"type": "expression", "latex": "x^2"},
            modality="text"
        )
        
        self.context.add_entity(
            entity_id="entity2",
            entity_data={"type": "expression", "latex": "x^2"},
            modality="image"
        )
        
        # Add reference
        reference = self.context.add_reference(
            source_id="entity1",
            target_id="entity2",
            ref_type="equivalent"
        )
        
        # Verify reference was added
        self.assertEqual(reference["source_id"], "entity1")
        self.assertEqual(reference["target_id"], "entity2")
        self.assertEqual(reference["type"], "equivalent")
        
        reference_id = "entity1_entity2_equivalent"
        self.assertIn(reference_id, self.context.references)
    
    def test_find_references(self):
        """Test finding references for an entity."""
        # Add entities
        self.context.add_entity(
            entity_id="entity1",
            entity_data={"type": "expression", "latex": "x^2"},
            modality="text"
        )
        
        self.context.add_entity(
            entity_id="entity2",
            entity_data={"type": "expression", "latex": "x^2"},
            modality="image"
        )
        
        self.context.add_entity(
            entity_id="entity3",
            entity_data={"type": "expression", "latex": "y^2"},
            modality="image"
        )
        
        # Add references
        self.context.add_reference(
            source_id="entity1",
            target_id="entity2",
            ref_type="equivalent"
        )
        
        self.context.add_reference(
            source_id="entity1",
            target_id="entity3",
            ref_type="related"
        )
        
        # Find references
        refs = self.context.find_references("entity1")
        self.assertEqual(len(refs), 2)
        
        # Find references by type
        equiv_refs = self.context.find_references("entity1", "equivalent")
        self.assertEqual(len(equiv_refs), 1)
        self.assertEqual(equiv_refs[0]["target_id"], "entity2")
    
    def test_to_dict_and_from_dict(self):
        """Test conversion to and from dictionary."""
        # Add some entities and references
        self.context.add_entity(
            entity_id="entity1",
            entity_data={"type": "expression", "latex": "x^2"},
            modality="text"
        )
        
        self.context.add_entity(
            entity_id="entity2",
            entity_data={"type": "expression", "latex": "x^2"},
            modality="image"
        )
        
        self.context.add_reference(
            source_id="entity1",
            target_id="entity2",
            ref_type="equivalent"
        )
        
        # Convert to dict
        context_dict = self.context.to_dict()
        
        # Create new context from dict
        new_context = CrossModalContext.from_dict(context_dict)
        
        # Verify data was preserved
        self.assertEqual(new_context.context_id, self.context.context_id)
        self.assertIn("entity1", new_context.entities)
        self.assertIn("entity2", new_context.entities)
        self.assertIn("entity1_entity2_equivalent", new_context.references)
        self.assertIn("entity1", new_context.modalities.get("text", []))
        self.assertIn("entity2", new_context.modalities.get("image", []))


class TestContextManager(unittest.TestCase):
    """Tests for the context manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = ContextManager()
    
    def test_create_context(self):
        """Test creating a new context."""
        # Create context
        context = self.manager.create_context()
        
        # Verify context was created
        self.assertIsNotNone(context)
        self.assertIn(context.context_id, self.manager.active_contexts)
    
    def test_get_context(self):
        """Test retrieving a context."""
        # Create context
        context = self.manager.create_context()
        
        # Retrieve context
        retrieved = self.manager.get_context(context.context_id)
        
        # Verify context was retrieved
        self.assertEqual(retrieved.context_id, context.context_id)
    
    def test_add_entity_to_context(self):
        """Test adding an entity to a context."""
        # Create context
        context = self.manager.create_context()
        
        # Add entity
        entity_id = self.manager.add_entity_to_context(
            context_id=context.context_id,
            entity_data={"type": "expression", "latex": "x^2"},
            modality="text"
        )
        
        # Verify entity was added
        self.assertIsNotNone(entity_id)
        
        # Get updated context
        updated_context = self.manager.get_context(context.context_id)
        self.assertIn(entity_id, updated_context.entities)
    
    def test_add_reference_to_context(self):
        """Test adding a reference to a context."""
        # Create context
        context = self.manager.create_context()
        
        # Add entities
        entity1_id = self.manager.add_entity_to_context(
            context_id=context.context_id,
            entity_data={"type": "expression", "latex": "x^2"},
            modality="text"
        )
        
        entity2_id = self.manager.add_entity_to_context(
            context_id=context.context_id,
            entity_data={"type": "expression", "latex": "x^2"},
            modality="image"
        )
        
        # Add reference
        reference = self.manager.add_reference_to_context(
            context_id=context.context_id,
            source_id=entity1_id,
            target_id=entity2_id,
            ref_type="equivalent"
        )
        
        # Verify reference was added
        self.assertIsNotNone(reference)
        
        # Get updated context
        updated_context = self.manager.get_context(context.context_id)
        refs = updated_context.find_references(entity1_id)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["target_id"], entity2_id)


class TestReferenceResolver(unittest.TestCase):
    """Tests for the reference resolver."""
    
    def setUp(self):
        """Set up test environment."""
        self.resolver = ReferenceResolver()
        self.context_manager = ContextManager()
        self.context = self.context_manager.create_context()
    
    def test_extract_references(self):
        """Test extracting references from text."""
        # Test text with references
        text = "As shown in equation 1, the derivative of x^2 is 2x. The above equation demonstrates this property."
        
        # Extract references
        references = self.resolver._extract_references(text)
        
        # Verify references were extracted
        self.assertEqual(len(references), 2)
        
        # Check first reference (explicit)
        self.assertEqual(references[0][0], "equation 1")
        self.assertEqual(references[0][1], "equation")
        self.assertEqual(references[0][2], "1")
        
        # Check second reference (positional)
        self.assertEqual(references[1][0], "above equation")
        self.assertEqual(references[1][1], "equation")
        self.assertEqual(references[1][2], "previous")
    
    def test_resolve_references(self):
        """Test resolving references in text."""
        # Add entities to context
        self.context_manager.add_entity_to_context(
            context_id=self.context.context_id,
            entity_data={
                "type": "equation",
                "latex": "\\frac{d}{dx}(x^2) = 2x",
                "number": 1
            },
            modality="text",
            entity_id="eq1"
        )
        
        # Text with references
        text = "As shown in equation 1, the derivative of x^2 is 2x."
        
        # Resolve references
        result = self.resolver.resolve_references(self.context, text)
        
        # Verify result
        self.assertEqual(result["original_text"], text)
        self.assertIn("equation 1", result["resolved_references"])
        self.assertEqual(result["resolved_references"]["equation 1"]["id"], "1")
    
    def test_calculate_latex_similarity(self):
        """Test calculating similarity between LaTeX expressions."""
        # Test exact match
        self.assertEqual(
            self.resolver._calculate_latex_similarity("x^2", "x^2"),
            1.0
        )
        
        # Test similar expressions
        similarity = self.resolver._calculate_latex_similarity("\\frac{x^2}{2}", "x^2/2")
        self.assertGreater(similarity, 0.5)
        
        # Test different expressions
        similarity = self.resolver._calculate_latex_similarity("x^2", "\\sin(x)")
        self.assertLess(similarity, 0.5)


if __name__ == '__main__':
    unittest.main()

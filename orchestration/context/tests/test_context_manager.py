"""
Unit tests for the context manager.
"""

import unittest
from unittest.mock import MagicMock, patch
import time

from orchestration.context.context_manager import ContextManager
from orchestration.context.conversation_state import ConversationState, Message
from orchestration.context.entity_tracker import EntityTracker
from orchestration.context.pruning_strategy import TokenBudgetStrategy


class TestContextManager(unittest.TestCase):
    """Tests for the ContextManager class."""
    
    def setUp(self):
        self.context_manager = ContextManager(
            max_context_tokens=1000,
            entity_tracking_enabled=True
        )
    
    def test_create_conversation(self):
        """Test creating a new conversation."""
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Verify conversation was created
        self.assertIn(conversation_id, self.context_manager.conversation_states)
        
        # Verify state was initialized correctly
        state = self.context_manager.conversation_states[conversation_id]
        self.assertEqual(state.user_id, user_id)
        self.assertEqual(state.get_message_count(), 0)
        
        # Verify entity tracker was created
        self.assertIn(conversation_id, self.context_manager.entity_trackers)
    
    def test_add_user_message(self):
        """Test adding a user message to a conversation."""
        # Create a conversation
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Add a user message
        message = "This is a test message"
        result = self.context_manager.add_user_message(conversation_id, message)
        
        # Verify message was added
        self.assertIn("message_id", result)
        state = self.context_manager.conversation_states[conversation_id]
        self.assertEqual(state.get_message_count(), 1)
        
        # Verify message content
        added_message = state.get_message(result["message_id"])
        self.assertEqual(added_message.role, "user")
        self.assertEqual(added_message.content, message)
    
    def test_add_system_message(self):
        """Test adding a system message to a conversation."""
        # Create a conversation
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Add a system message
        message = "This is a system message"
        result = self.context_manager.add_system_message(conversation_id, message)
        
        # Verify message was added
        self.assertIn("message_id", result)
        state = self.context_manager.conversation_states[conversation_id]
        self.assertEqual(state.get_message_count(), 1)
        
        # Verify message content
        added_message = state.get_message(result["message_id"])
        self.assertEqual(added_message.role, "system")
        self.assertEqual(added_message.content, message)
    
    def test_get_conversation_context(self):
        """Test getting conversation context."""
        # Create a conversation
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Add messages
        self.context_manager.add_user_message(conversation_id, "User message 1")
        self.context_manager.add_system_message(conversation_id, "System message 1")
        self.context_manager.add_user_message(conversation_id, "User message 2")
        self.context_manager.add_system_message(conversation_id, "System message 2")
        
        # Get context in different formats
        text_context = self.context_manager.get_conversation_context(
            conversation_id, format="text")
        dict_context = self.context_manager.get_conversation_context(
            conversation_id, format="dict")
        json_context = self.context_manager.get_conversation_context(
            conversation_id, format="json")
        
        # Verify text context
        self.assertIn("USER: User message 1", text_context)
        self.assertIn("SYSTEM: System message 1", text_context)
        self.assertIn("USER: User message 2", text_context)
        self.assertIn("SYSTEM: System message 2", text_context)
        
        # Verify dict context
        self.assertEqual(dict_context["conversation_id"], conversation_id)
        self.assertEqual(dict_context["user_id"], user_id)
        self.assertEqual(len(dict_context["messages"]), 4)
        
        # Verify json context (basic check)
        self.assertIsInstance(json_context, str)
        self.assertIn(conversation_id, json_context)
    
    def test_resolve_entity_references(self):
        """Test resolving entity references."""
        # Create a conversation
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Add messages with mathematical content
        self.context_manager.add_user_message(
            conversation_id, "Let f(x) = x^2 + 2x + 1")
        self.context_manager.add_system_message(
            conversation_id, "I've defined the function f(x) = x^2 + 2x + 1.")
        
        # Test reference resolution
        query = "Find the derivative of this function"
        result = self.context_manager.resolve_entity_references(conversation_id, query)
        
        # Verify references were resolved
        self.assertIsInstance(result, dict)
        self.assertIn("resolved_query", result)
        self.assertIn("referenced_entities", result)
    
    def test_get_relevant_entities(self):
        """Test getting relevant entities."""
        # Create a conversation
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Add messages with mathematical content
        self.context_manager.add_user_message(
            conversation_id, "Let f(x) = x^2 + 2x + 1")
        self.context_manager.add_system_message(
            conversation_id, "I've defined the function f(x) = x^2 + 2x + 1.")
        self.context_manager.add_user_message(
            conversation_id, "Let g(x) = sin(x)")
        self.context_manager.add_system_message(
            conversation_id, "I've defined the function g(x) = sin(x).")
        
        # Test getting relevant entities
        query = "Find the derivative of f(x)"
        relevant_entities = self.context_manager.get_relevant_entities(
            conversation_id, query, max_entities=3)
        
        # Verify relevant entities were found
        self.assertIsInstance(relevant_entities, list)
        self.assertGreaterEqual(len(relevant_entities), 1)
    
    def test_clear_conversation(self):
        """Test clearing a conversation."""
        # Create a conversation
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Add messages
        self.context_manager.add_user_message(conversation_id, "User message 1")
        self.context_manager.add_system_message(conversation_id, "System message 1")
        
        # Clear the conversation
        result = self.context_manager.clear_conversation(conversation_id)
        
        # Verify conversation was cleared
        self.assertTrue(result)
        state = self.context_manager.conversation_states[conversation_id]
        self.assertEqual(state.get_message_count(), 0)
        
        # Verify metadata was preserved
        self.assertEqual(state.user_id, user_id)
    
    def test_delete_conversation(self):
        """Test deleting a conversation."""
        # Create a conversation
        user_id = "test_user"
        conversation_id = self.context_manager.create_conversation(user_id)
        
        # Add messages
        self.context_manager.add_user_message(conversation_id, "User message 1")
        self.context_manager.add_system_message(conversation_id, "System message 1")
        
        # Delete the conversation
        result = self.context_manager.delete_conversation(conversation_id)
        
        # Verify conversation was deleted
        self.assertTrue(result)
        self.assertNotIn(conversation_id, self.context_manager.conversation_states)
        self.assertNotIn(conversation_id, self.context_manager.entity_trackers)
    
    def test_apply_pruning(self):
        """Test pruning is applied when context exceeds limits."""
        # Create a pruning strategy mock
        pruning_strategy = MagicMock()
        
        # Create a context manager with the mock strategy
        context_manager = ContextManager(
            max_context_tokens=100,
            entity_tracking_enabled=True,
            pruning_strategy=pruning_strategy
        )
        
        # Create a conversation
        user_id = "test_user"
        conversation_id = context_manager.create_conversation(user_id)
        
        # Mock the token count to exceed the limit
        state = context_manager.conversation_states[conversation_id]
        state.estimate_token_count = MagicMock(return_value=150)
        
        # Add a message to trigger pruning
        context_manager.add_user_message(conversation_id, "Test message")
        
        # Verify pruning strategy was applied
        pruning_strategy.apply.assert_called_once_with(state)


class TestConversationState(unittest.TestCase):
    """Tests for the ConversationState class."""
    
    def setUp(self):
        self.conversation_id = "test_conversation"
        self.user_id = "test_user"
        self.state = ConversationState(self.conversation_id, self.user_id)
    
    def test_add_user_message(self):
        """Test adding a user message."""
        content = "Test user message"
        message_id = self.state.add_user_message(content)
        
        # Verify message was added
        self.assertEqual(self.state.get_message_count(), 1)
        message = self.state.get_message(message_id)
        self.assertIsNotNone(message)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, content)
    
    def test_add_system_message(self):
        """Test adding a system message."""
        content = "Test system message"
        message_id = self.state.add_system_message(content)
        
        # Verify message was added
        self.assertEqual(self.state.get_message_count(), 1)
        message = self.state.get_message(message_id)
        self.assertIsNotNone(message)
        self.assertEqual(message.role, "system")
        self.assertEqual(message.content, content)
    
    def test_remove_message(self):
        """Test removing a message."""
        # Add a message
        message_id = self.state.add_user_message("Test message")
        
        # Verify it was added
        self.assertEqual(self.state.get_message_count(), 1)
        
        # Remove the message
        result = self.state.remove_message(message_id)
        
        # Verify it was removed
        self.assertTrue(result)
        self.assertEqual(self.state.get_message_count(), 0)
        self.assertIsNone(self.state.get_message(message_id))
    
    def test_update_message_metadata(self):
        """Test updating message metadata."""
        # Add a message
        message_id = self.state.add_user_message("Test message")
        
        # Update metadata
        updates = {"key1": "value1", "key2": 42}
        result = self.state.update_message_metadata(message_id, updates)
        
        # Verify update was successful
        self.assertTrue(result)
        
        # Verify metadata was updated
        message = self.state.get_message(message_id)
        self.assertEqual(message.metadata["key1"], "value1")
        self.assertEqual(message.metadata["key2"], 42)
    
    def test_estimate_token_count(self):
        """Test estimating token count."""
        # Empty conversation
        empty_count = self.state.estimate_token_count()
        
        # Add messages
        self.state.add_user_message("Test user message")
        self.state.add_system_message("Test system message")
        
        # Estimate with messages
        message_count = self.state.estimate_token_count()
        
        # Verify counts
        self.assertGreater(message_count, empty_count)
    
    def test_get_context_text(self):
        """Test getting context as text."""
        # Add messages
        self.state.add_user_message("User message 1")
        self.state.add_system_message("System message 1")
        self.state.add_user_message("User message 2")
        self.state.add_system_message("System message 2")
        
        # Get context text
        context_text = self.state.get_context_text()
        
        # Verify text format
        self.assertIn("USER: User message 1", context_text)
        self.assertIn("SYSTEM: System message 1", context_text)
        self.assertIn("USER: User message 2", context_text)
        self.assertIn("SYSTEM: System message 2", context_text)
    
    def test_get_context_dict(self):
        """Test getting context as a dictionary."""
        # Add messages
        self.state.add_user_message("User message 1")
        self.state.add_system_message("System message 1")
        
        # Get context dict
        context_dict = self.state.get_context_dict()
        
        # Verify dict format
        self.assertEqual(context_dict["conversation_id"], self.conversation_id)
        self.assertEqual(context_dict["user_id"], self.user_id)
        self.assertEqual(len(context_dict["messages"]), 2)
        
        # Verify message format
        message = context_dict["messages"][0]
        self.assertIn("message_id", message)
        self.assertIn("role", message)
        self.assertIn("content", message)
        self.assertIn("timestamp", message)
    
    def test_token_budget_handling(self):
        """Test handling token budget constraints."""
        # Set a max token parameter
        max_tokens = 20
        
        # Add enough messages to exceed the budget
        for i in range(10):
            self.state.add_user_message(f"Message {i}")
        
        # Get context within token budget
        limited_context = self.state.get_context_text(max_tokens=max_tokens)
        
        # Verify we only got some messages
        message_count = limited_context.count("USER:")
        self.assertLess(message_count, 10)


class TestEntityTracker(unittest.TestCase):
    """Tests for the EntityTracker class."""
    
    def setUp(self):
        self.tracker = EntityTracker()
    
    def test_extract_latex_expressions(self):
        """Test extracting LaTeX expressions."""
        text = "Consider the equation $x^2 + 2x + 1 = 0$ and also $$\\int_0^1 x^2 dx$$"
        expressions = self.tracker._extract_latex_expressions(text)
        
        # Verify expressions were extracted
        self.assertEqual(len(expressions), 2)
        self.assertIn("x^2 + 2x + 1 = 0", expressions)
        self.assertIn("\\int_0^1 x^2 dx", expressions)
    
    def test_extract_variable_definitions(self):
        """Test extracting variable definitions."""
        text = "Let x = 5 and define y as 2x + 3. Also, z is defined as x + y."
        variables = self.tracker._extract_variable_definitions(text)
        
        # Verify variables were extracted
        self.assertEqual(len(variables), 3)
        self.assertIn(("x", "5"), variables)
        self.assertIn(("y", "2x + 3"), variables)
        self.assertIn(("z", "x + y"), variables)
    
    def test_extract_function_definitions(self):
        """Test extracting function definitions."""
        text = "Let f(x) = x^2 + 1 and define g(x, y) as x + y."
        functions = self.tracker._extract_function_definitions(text)
        
        # Verify functions were extracted
        self.assertEqual(len(functions), 2)
        self.assertTrue(any(f[0] == "f" for f in functions))
        self.assertTrue(any(f[0] == "g" for f in functions))
    
    def test_extract_entities(self):
        """Test extracting mathematical entities."""
        text = "Let f(x) = x^2 + 2x + 1 and $y = \\sqrt{x}$."
        entities = self.tracker.extract_entities(text)
        
        # Verify entities were extracted
        self.assertGreaterEqual(len(entities), 2)  # At least f(x) and y
    
    def test_resolve_references(self):
        """Test resolving references to entities."""
        # Add some entities
        self.tracker._add_entity("function", "x^2 + 1", display_form="f(x)")
        self.tracker._add_entity("variable", "5", display_form="a")
        
        # Test reference resolution
        query = "Find the derivative of the function"
        resolved_query, referenced_entities = self.tracker.resolve_references(query)
        
        # Verify reference was resolved
        self.assertNotEqual(resolved_query, query)
        self.assertGreaterEqual(len(referenced_entities), 1)
    
    def test_update_entity(self):
        """Test updating an entity."""
        # Add an entity
        entity_id = self.tracker._add_entity("variable", "5", display_form="x")
        
        # Update the entity
        updates = {"value": "10", "display_form": "x_new"}
        result = self.tracker.update_entity(entity_id, updates)
        
        # Verify update was successful
        self.assertTrue(result)
        
        # Verify entity was updated
        entity_dict = self.tracker.get_entity(entity_id)
        self.assertEqual(entity_dict["value"], "10")
        self.assertEqual(entity_dict["display_form"], "x_new")


class TestPruningStrategies(unittest.TestCase):
    """Tests for pruning strategies."""
    
    def setUp(self):
        self.conversation_id = "test_conversation"
        self.user_id = "test_user"
        self.state = ConversationState(self.conversation_id, self.user_id)
        
        # Add some messages
        for i in range(10):
            if i % 2 == 0:
                self.state.add_user_message(f"User message {i}")
            else:
                self.state.add_system_message(f"System message {i}")
    
    def test_token_budget_strategy(self):
        """Test the TokenBudgetStrategy."""
        # Create a strategy with a low token budget
        strategy = TokenBudgetStrategy(
            max_tokens=50,
            target_ratio=0.8,
            preserve_last_n_turns=1
        )
        
        # Get initial message count
        initial_count = self.state.get_message_count()
        
        # Apply the strategy
        strategy.apply(self.state)
        
        # Verify messages were pruned
        final_count = self.state.get_message_count()
        self.assertLess(final_count, initial_count)
        
        # Verify most recent turn was preserved
        messages = self.state.messages
        self.assertEqual(messages[-1].role, "system")
        self.assertEqual(messages[-2].role, "user")
    
    def test_relevance_pruning_strategy(self):
        """Test the RelevancePruningStrategy."""
        # Create a strategy with a low token budget
        strategy = RelevancePruningStrategy(
            max_tokens=50,
            target_ratio=0.8,
            preserve_last_n_turns=1
        )
        
        # Get initial message count
        initial_count = self.state.get_message_count()
        
        # Apply the strategy
        strategy.apply(self.state)
        
        # Verify messages were pruned
        final_count = self.state.get_message_count()
        self.assertLess(final_count, initial_count)
        
        # Verify most recent turn was preserved
        messages = self.state.messages
        self.assertEqual(messages[-1].role, "system")
        self.assertEqual(messages[-2].role, "user")


if __name__ == "__main__":
    unittest.main()

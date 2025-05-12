# Context Management System

The Context Management system for the Mathematical Multimodal LLM System provides robust handling of conversation state, mathematical entity tracking, and context pruning. This is a critical component that enables coherent multi-turn conversations, particularly for mathematical discussions where precise tracking of mathematical entities and references is essential.

## Key Components

### ContextManager

The central component that coordinates all context management functionality:

- Maintains conversation states
- Tracks mathematical entities
- Resolves references to entities
- Applies pruning strategies when context exceeds limits
- Provides context information in various formats

### ConversationState

Manages the state of a conversation:

- Stores messages with metadata
- Handles token counting
- Provides context in text or dictionary format
- Manages removal of messages when needed

### EntityTracker

Specialized component for mathematical entities:

- Extracts mathematical entities from text (expressions, variables, functions)
- Resolves references to entities in subsequent messages
- Tracks entity usage and allows updating entity information
- Finds entities relevant to a given query

### Pruning Strategies

Strategies for managing context size:

- **TokenBudgetStrategy**: Removes oldest messages first to stay within token limits
- **RelevancePruningStrategy**: Removes less relevant messages based on current context
- **SummaryPruningStrategy**: Replaces older message groups with summaries

## Usage Examples

### Basic Conversation Context

```python
from orchestration.context.context_manager import get_context_manager

# Get the context manager singleton
context_manager = get_context_manager()

# Create a new conversation
user_id = "user123"
conversation_id = context_manager.create_conversation(user_id)

# Add messages to the conversation
context_manager.add_user_message(conversation_id, "Let f(x) = x^2 + 3x + 2")
context_manager.add_system_message(
    conversation_id,
    "I've defined the function f(x) = x^2 + 3x + 2."
)

# Get the conversation context
context = context_manager.get_conversation_context(conversation_id)
```

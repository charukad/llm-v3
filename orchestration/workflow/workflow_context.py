"""
Workflow Context Manager for the Mathematical Multimodal LLM System.

This module provides context management for workflow executions,
including state persistence, variable management, and execution history.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class WorkflowContext(BaseModel):
    """Manages context and state for workflow executions."""
    
    workflow_id: str
    execution_id: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a variable in the workflow context."""
        self.variables[key] = value
        self.updated_at = datetime.utcnow()
        self._log_state_change("variable_set", key, value)
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the workflow context."""
        return self.variables.get(key, default)
    
    def update_state(self, key: str, value: Any) -> None:
        """Update the workflow state."""
        self.state[key] = value
        self.updated_at = datetime.utcnow()
        self._log_state_change("state_update", key, value)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value from the workflow context."""
        return self.state.get(key, default)
    
    def add_history_entry(self, event_type: str, details: Dict[str, Any]) -> None:
        """Add an entry to the workflow history."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.history.append(entry)
        self.updated_at = datetime.utcnow()
    
    def _log_state_change(self, change_type: str, key: str, value: Any) -> None:
        """Log a state change to the history."""
        self.add_history_entry(
            change_type,
            {
                "key": key,
                "value": value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "variables": self.variables,
            "state": self.state,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowContext':
        """Create a context from a dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            execution_id=data["execution_id"],
            variables=data.get("variables", {}),
            state=data.get("state", {}),
            history=data.get("history", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        ) 
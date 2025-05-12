"""
Workflow definition for the Mathematical Multimodal LLM System.

This module defines the structure of workflows and steps for orchestration.
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class WorkflowStep:
    """
    Definition of a step in a workflow.
    
    A step represents a single operation that needs to be executed as part of
    a workflow, specifying the required capability and input/output keys.
    """
    
    def __init__(
        self,
        id: str,
        required_capability: str,
        message_type: str,
        input_keys: List[str],
        output_keys: List[str],
        description: Optional[str] = None,
        retry_policy: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a workflow step.
        
        Args:
            id: Step ID
            required_capability: Required capability for agent selection
            message_type: Message type for the step
            input_keys: Keys required from workflow data
            output_keys: Keys to be added to workflow data
            description: Optional step description
            retry_policy: Optional retry policy for failures
        """
        self.id = id
        self.required_capability = required_capability
        self.message_type = message_type
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.description = description or f"Step {id} requiring {required_capability}"
        
        # Set default retry policy if not provided
        self.retry_policy = retry_policy or {
            "max_retries": 3,
            "initial_delay": 1,
            "max_delay": 30,
            "backoff_factor": 2,
            "jitter": 0.1
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the step to a dictionary.
        
        Returns:
            Dictionary representation of the step
        """
        return {
            "id": self.id,
            "required_capability": self.required_capability,
            "message_type": self.message_type,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "description": self.description,
            "retry_policy": self.retry_policy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowStep':
        """
        Create a step from a dictionary.
        
        Args:
            data: Dictionary representation of the step
            
        Returns:
            WorkflowStep object
        """
        return cls(
            id=data["id"],
            required_capability=data["required_capability"],
            message_type=data["message_type"],
            input_keys=data["input_keys"],
            output_keys=data["output_keys"],
            description=data.get("description"),
            retry_policy=data.get("retry_policy")
        )

class WorkflowDefinition:
    """
    Definition of a workflow.
    
    A workflow consists of a sequence of steps that need to be executed
    in order to complete a particular task.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        steps: List[WorkflowStep],
        timeout: Optional[int] = None,
        version: str = "1.0.0"
    ):
        """
        Initialize a workflow definition.
        
        Args:
            id: Workflow ID
            name: Workflow name
            description: Workflow description
            steps: List of workflow steps
            timeout: Optional timeout in seconds
            version: Workflow definition version
        """
        self.id = id
        self.name = name
        self.description = description
        self.steps = steps
        self.timeout = timeout or 300  # Default timeout: 5 minutes
        self.version = version
    
    def get_input_keys(self) -> List[str]:
        """
        Get all input keys required by this workflow.
        
        Returns:
            List of required input keys
        """
        input_keys = set()
        for step in self.steps:
            for key in step.input_keys:
                input_keys.add(key)
        
        return list(input_keys)
    
    def get_output_keys(self) -> List[str]:
        """
        Get all output keys produced by this workflow.
        
        Returns:
            List of output keys
        """
        output_keys = set()
        for step in self.steps:
            for key in step.output_keys:
                output_keys.add(key)
        
        return list(output_keys)
    
    def validate(self) -> bool:
        """
        Validate the workflow definition.
        
        Checks for common issues like missing inputs for steps.
        
        Returns:
            True if the workflow is valid, False otherwise
        """
        # Check for empty steps
        if not self.steps:
            logger.error(f"Workflow {self.id} has no steps")
            return False
        
        # Track available keys as workflow progresses
        available_keys = set()
        
        # Check each step
        for i, step in enumerate(self.steps):
            # Check for missing input keys
            missing_keys = [key for key in step.input_keys if key not in available_keys]
            
            if missing_keys:
                # First step can have external inputs
                if i > 0:
                    logger.error(f"Step {step.id} requires missing keys: {missing_keys}")
                    return False
            
            # Add output keys to available keys
            for key in step.output_keys:
                available_keys.add(key)
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow to a dictionary.
        
        Returns:
            Dictionary representation of the workflow
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "timeout": self.timeout,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowDefinition':
        """
        Create a workflow from a dictionary.
        
        Args:
            data: Dictionary representation of the workflow
            
        Returns:
            WorkflowDefinition object
        """
        steps = [WorkflowStep.from_dict(step_data) for step_data in data["steps"]]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=steps,
            timeout=data.get("timeout"),
            version=data.get("version", "1.0.0")
        )

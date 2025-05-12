"""
Error Recovery Mechanisms for the Mathematical Multimodal LLM System.

This module provides advanced error recovery strategies for handling failures
in mathematical workflows, including fallback mechanisms, compensation strategies,
and graceful degradation options.
"""
import asyncio
import datetime
import json
from typing import Dict, Any, List, Optional, Set, Tuple, Callable, Union
import logging
import traceback
import copy
import random

from ..monitoring.logger import get_logger
from ..monitoring.tracing import get_tracer
from .workflow_engine import get_workflow_engine, WorkflowExecution, ActivityStatus, WorkflowExecutionStatus

logger = get_logger(__name__)


class ErrorCategory:
    """Error categories for classification."""
    COMPUTATION = "computation"
    VISUALIZATION = "visualization"
    OCR = "ocr"
    SEARCH = "search"
    COMMUNICATION = "communication"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    VALIDATION = "validation"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


class ErrorSeverity:
    """Severity levels for errors."""
    CRITICAL = "critical"  # Workflow cannot continue, needs manual intervention
    HIGH = "high"          # Can potentially continue with significant degradation
    MEDIUM = "medium"      # Can continue with some degradation
    LOW = "low"            # Can be handled automatically with minimal impact


class ErrorClassification:
    """
    Classification of an error for recovery purposes.
    
    This class represents a structured classification of an error,
    including its category, severity, and recovery options.
    """
    
    def __init__(
        self,
        error_message: str,
        error_code: str,
        category: str = ErrorCategory.UNKNOWN,
        severity: str = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        details: Dict[str, Any] = None
    ):
        """
        Initialize an error classification.
        
        Args:
            error_message: Human-readable error message
            error_code: Machine-readable error code
            category: Error category
            severity: Error severity
            recoverable: Whether the error is recoverable
            details: Additional error details
        """
        self.error_message = error_message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.details = details or {}
        self.timestamp = datetime.datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "error_message": self.error_message,
            "error_code": self.error_code,
            "category": self.category,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "details": self.details,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorClassification':
        """Create from dictionary representation."""
        return cls(
            error_message=data["error_message"],
            error_code=data["error_code"],
            category=data.get("category", ErrorCategory.UNKNOWN),
            severity=data.get("severity", ErrorSeverity.MEDIUM),
            recoverable=data.get("recoverable", True),
            details=data.get("details", {})
        )
        
    @classmethod
    def from_exception(cls, exception: Exception, category: str = None) -> 'ErrorClassification':
        """
        Create an error classification from an exception.
        
        Args:
            exception: The exception to classify
            category: Optional explicit category
            
        Returns:
            ErrorClassification instance
        """
        # Extract error message
        error_message = str(exception)
        
        # Determine error code based on exception type
        error_code = type(exception).__name__
        
        # Guess category from exception if not provided
        if not category:
            if isinstance(exception, asyncio.TimeoutError):
                category = ErrorCategory.TIMEOUT
            elif isinstance(exception, (ValueError, TypeError)):
                category = ErrorCategory.VALIDATION
            elif isinstance(exception, PermissionError):
                category = ErrorCategory.PERMISSION
            elif isinstance(exception, (ConnectionError, OSError)):
                category = ErrorCategory.COMMUNICATION
            elif isinstance(exception, MemoryError):
                category = ErrorCategory.RESOURCE
            else:
                category = ErrorCategory.UNKNOWN
                
        # Determine severity
        severity = ErrorSeverity.MEDIUM
        if category in [ErrorCategory.PERMISSION, ErrorCategory.RESOURCE]:
            severity = ErrorSeverity.HIGH
        elif category == ErrorCategory.TIMEOUT:
            severity = ErrorSeverity.MEDIUM
        elif category == ErrorCategory.VALIDATION:
            severity = ErrorSeverity.LOW
            
        # Collect details
        details = {
            "exception_type": error_code,
            "traceback": traceback.format_exc()
        }
        
        return cls(
            error_message=error_message,
            error_code=error_code,
            category=category,
            severity=severity,
            recoverable=True,  # Assume recoverable by default
            details=details
        )


class RecoveryStrategy:
    """
    A strategy for recovering from an error.
    
    This class represents a structured approach to recovering from
    an error, including fallback options and retry parameters.
    """
    
    def __init__(
        self,
        strategy_type: str,
        params: Dict[str, Any] = None,
        description: str = None,
        fallback_strategies: List[str] = None
    ):
        """
        Initialize a recovery strategy.
        
        Args:
            strategy_type: Type of recovery strategy
            params: Parameters for the strategy
            description: Human-readable description
            fallback_strategies: Next strategies to try if this one fails
        """
        self.strategy_type = strategy_type
        self.params = params or {}
        self.description = description or f"{strategy_type} recovery strategy"
        self.fallback_strategies = fallback_strategies or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy_type": self.strategy_type,
            "params": self.params,
            "description": self.description,
            "fallback_strategies": self.fallback_strategies
        }
        
    @classmethod
    def retry_strategy(
        cls,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        jitter: float = 0.2,
        timeout_multiplier: float = 1.0
    ) -> 'RecoveryStrategy':
        """
        Create a retry strategy.
        
        Args:
            max_retries: Maximum number of retries
            backoff_factor: Factor for exponential backoff
            jitter: Jitter factor for backoff
            timeout_multiplier: Multiplier for timeout on subsequent retries
            
        Returns:
            RecoveryStrategy for retrying
        """
        return cls(
            strategy_type="retry",
            params={
                "max_retries": max_retries,
                "backoff_factor": backoff_factor,
                "jitter": jitter,
                "timeout_multiplier": timeout_multiplier
            },
            description=f"Retry up to {max_retries} times with exponential backoff",
            fallback_strategies=["simplify", "approximate"]
        )
        
    @classmethod
    def simplify_strategy(
        cls,
        simplification_level: int = 1,
        preserve_core_functionality: bool = True
    ) -> 'RecoveryStrategy':
        """
        Create a simplification strategy.
        
        Args:
            simplification_level: Level of simplification (1-3)
            preserve_core_functionality: Whether to preserve core functionality
            
        Returns:
            RecoveryStrategy for simplification
        """
        return cls(
            strategy_type="simplify",
            params={
                "simplification_level": simplification_level,
                "preserve_core_functionality": preserve_core_functionality
            },
            description=f"Simplify the operation to level {simplification_level}",
            fallback_strategies=["approximate", "skip"]
        )
        
    @classmethod
    def approximate_strategy(
        cls,
        precision: str = "medium",
        numerical_fallback: bool = True
    ) -> 'RecoveryStrategy':
        """
        Create an approximation strategy.
        
        Args:
            precision: Precision level (high, medium, low)
            numerical_fallback: Whether to use numerical methods as fallback
            
        Returns:
            RecoveryStrategy for approximation
        """
        return cls(
            strategy_type="approximate",
            params={
                "precision": precision,
                "numerical_fallback": numerical_fallback
            },
            description=f"Use approximate computation with {precision} precision",
            fallback_strategies=["skip"]
        )
        
    @classmethod
    def alternative_agent_strategy(
        cls,
        capability: str,
        preferred_types: List[str] = None
    ) -> 'RecoveryStrategy':
        """
        Create a strategy to use an alternative agent.
        
        Args:
            capability: Required capability
            preferred_types: Preferred agent types
            
        Returns:
            RecoveryStrategy for using alternative agents
        """
        return cls(
            strategy_type="alternative_agent",
            params={
                "capability": capability,
                "preferred_types": preferred_types or []
            },
            description=f"Use an alternative agent with {capability} capability",
            fallback_strategies=["retry", "simplify"]
        )
        
    @classmethod
    def skip_strategy(
        cls,
        provide_explanation: bool = True
    ) -> 'RecoveryStrategy':
        """
        Create a strategy to skip the failed operation.
        
        Args:
            provide_explanation: Whether to provide an explanation for skipping
            
        Returns:
            RecoveryStrategy for skipping
        """
        return cls(
            strategy_type="skip",
            params={
                "provide_explanation": provide_explanation
            },
            description="Skip the failed operation and continue",
            fallback_strategies=["minimal_response"]
        )
        
    @classmethod
    def minimal_response_strategy(cls) -> 'RecoveryStrategy':
        """
        Create a strategy to provide a minimal response.
        
        Returns:
            RecoveryStrategy for minimal response
        """
        return cls(
            strategy_type="minimal_response",
            params={},
            description="Provide a minimal response without the failed operation",
            fallback_strategies=[]
        )


class RecoveryAction:
    """
    A concrete action to recover from an error.
    
    This class represents a specific action to be taken to recover from
    an error, based on a recovery strategy.
    """
    
    def __init__(
        self,
        action_type: str,
        params: Dict[str, Any],
        error: ErrorClassification,
        workflow_id: str,
        activity_index: int,
        attempt: int = 1
    ):
        """
        Initialize a recovery action.
        
        Args:
            action_type: Type of recovery action
            params: Parameters for the action
            error: The error to recover from
            workflow_id: ID of the affected workflow
            activity_index: Index of the affected activity
            attempt: Current attempt number
        """
        self.action_type = action_type
        self.params = params
        self.error = error
        self.workflow_id = workflow_id
        self.activity_index = activity_index
        self.attempt = attempt
        self.created_at = datetime.datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action_type": self.action_type,
            "params": self.params,
            "error": self.error.to_dict() if isinstance(self.error, ErrorClassification) else self.error,
            "workflow_id": self.workflow_id,
            "activity_index": self.activity_index,
            "attempt": self.attempt,
            "created_at": self.created_at
        }


class RecoveryManager:
    """
    Manager for error recovery operations.
    
    This class provides functionality for determining and executing
    recovery strategies for various error scenarios.
    """
    
    def __init__(self):
        """Initialize the recovery manager."""
        self.workflow_engine = get_workflow_engine()
        self.tracer = get_tracer()
        self.recovery_handlers: Dict[str, List[Callable]] = {}
        self.strategy_implementations: Dict[str, Callable] = {}
        
        # Register default strategy implementations
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default strategy implementations."""
        self.register_strategy_implementation("retry", self._implement_retry_strategy)
        self.register_strategy_implementation("simplify", self._implement_simplify_strategy)
        self.register_strategy_implementation("approximate", self._implement_approximate_strategy)
        self.register_strategy_implementation("alternative_agent", self._implement_alternative_agent_strategy)
        self.register_strategy_implementation("skip", self._implement_skip_strategy)
        self.register_strategy_implementation("minimal_response", self._implement_minimal_response_strategy)
        
    def register_recovery_handler(self, error_category: str, handler: Callable):
        """
        Register a recovery handler for an error category.
        
        Args:
            error_category: Category of errors to handle
            handler: Async function that takes (error, workflow, activity_index)
        """
        if error_category not in self.recovery_handlers:
            self.recovery_handlers[error_category] = []
        self.recovery_handlers[error_category].append(handler)
        
    def register_strategy_implementation(self, strategy_type: str, implementation: Callable):
        """
        Register an implementation for a recovery strategy.
        
        Args:
            strategy_type: Type of strategy
            implementation: Async function that takes (strategy, workflow, activity_index)
        """
        self.strategy_implementations[strategy_type] = implementation
        
    def get_recovery_strategies(
        self,
        error: Union[ErrorClassification, Dict[str, Any]],
        domain: str = None
    ) -> List[RecoveryStrategy]:
        """
        Get appropriate recovery strategies for an error.
        
        Args:
            error: The error to recover from
            domain: Optional mathematical domain for context
            
        Returns:
            List of recovery strategies in priority order
        """
        # Convert dict to ErrorClassification if needed
        if isinstance(error, dict):
            error = ErrorClassification.from_dict(error)
            
        # Get appropriate strategies based on error category and severity
        strategies = []
        
        if error.severity == ErrorSeverity.CRITICAL:
            # Critical errors have limited recovery options
            strategies = [
                RecoveryStrategy.minimal_response_strategy()
            ]
        elif error.category == ErrorCategory.COMPUTATION:
            # Computation errors
            strategies = [
                RecoveryStrategy.retry_strategy(max_retries=2),
                RecoveryStrategy.simplify_strategy(simplification_level=1),
                RecoveryStrategy.approximate_strategy()
            ]
            
            # Domain-specific strategies
            if domain == "calculus":
                strategies.append(RecoveryStrategy(
                    strategy_type="numerical_integration",
                    params={"method": "adaptive"},
                    description="Use numerical integration instead of symbolic"
                ))
                
        elif error.category == ErrorCategory.VISUALIZATION:
            # Visualization errors
            strategies = [
                RecoveryStrategy.retry_strategy(max_retries=1),
                RecoveryStrategy.simplify_strategy(simplification_level=2),
                RecoveryStrategy.skip_strategy()
            ]
            
        elif error.category == ErrorCategory.OCR:
            # OCR errors
            strategies = [
                RecoveryStrategy.retry_strategy(max_retries=2),
                RecoveryStrategy(
                    strategy_type="enhance_image",
                    params={"methods": ["denoise", "sharpen", "contrast"]},
                    description="Enhance image quality",
                    fallback_strategies=["manual_input"]
                ),
                RecoveryStrategy.skip_strategy()
            ]
            
        elif error.category == ErrorCategory.TIMEOUT:
            # Timeout errors
            strategies = [
                RecoveryStrategy.retry_strategy(max_retries=1, timeout_multiplier=1.5),
                RecoveryStrategy.simplify_strategy(simplification_level=2),
                RecoveryStrategy.skip_strategy()
            ]
            
        elif error.category == ErrorCategory.RESOURCE:
            # Resource errors
            strategies = [
                RecoveryStrategy.simplify_strategy(simplification_level=3),
                RecoveryStrategy(
                    strategy_type="reduce_precision",
                    params={"target_precision": "single"},
                    description="Reduce computational precision"
                ),
                RecoveryStrategy.skip_strategy()
            ]
            
        else:
            # Default strategies for other error categories
            strategies = [
                RecoveryStrategy.retry_strategy(),
                RecoveryStrategy.alternative_agent_strategy(capability="compute"),
                RecoveryStrategy.skip_strategy()
            ]
            
        return strategies
        
    async def recover_from_error(
        self,
        error: Union[ErrorClassification, Dict[str, Any]],
        workflow_id: str,
        activity_index: int
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Attempt to recover from an error.
        
        Args:
            error: The error to recover from
            workflow_id: ID of the affected workflow
            activity_index: Index of the affected activity
            
        Returns:
            Tuple of (success, recovery_action)
        """
        # Convert dict to ErrorClassification if needed
        if isinstance(error, dict):
            error = ErrorClassification.from_dict(error)
            
        # Get the workflow
        workflow = await self.workflow_engine.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Cannot recover from error: workflow {workflow_id} not found")
            return False, None
            
        # Check if the activity exists
        if activity_index < 0 or activity_index >= len(workflow.activities):
            logger.error(f"Cannot recover from error: activity index {activity_index} out of range")
            return False, None
            
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Check current attempt number
        current_attempt = activity.get("attempt", 0) + 1
        
        # Get recovery strategies
        domain = workflow.context.get("domain", "general")
        strategies = self.get_recovery_strategies(error, domain)
        
        # Try recovery handlers first
        if error.category in self.recovery_handlers:
            for handler in self.recovery_handlers[error.category]:
                try:
                    success, action = await handler(error, workflow, activity_index)
                    if success:
                        return True, action
                except Exception as e:
                    logger.error(f"Error in recovery handler: {str(e)}")
                    
        # Try strategies in order
        for strategy in strategies:
            # Create recovery action
            action = RecoveryAction(
                action_type=strategy.strategy_type,
                params=strategy.params,
                error=error,
                workflow_id=workflow_id,
                activity_index=activity_index,
                attempt=current_attempt
            )
            
            # Try to implement the strategy
            if strategy.strategy_type in self.strategy_implementations:
                implementation = self.strategy_implementations[strategy.strategy_type]
                
                try:
                    success = await implementation(strategy, workflow, activity_index, action)
                    if success:
                        return True, action
                except Exception as e:
                    logger.error(f"Error implementing recovery strategy {strategy.strategy_type}: {str(e)}")
                    
        # No successful recovery
        logger.warning(f"All recovery strategies failed for workflow {workflow_id}, activity {activity_index}")
        return False, None
        
    async def _implement_retry_strategy(
        self,
        strategy: RecoveryStrategy,
        workflow: WorkflowExecution,
        activity_index: int,
        action: RecoveryAction
    ) -> bool:
        """
        Implement a retry strategy.
        
        Args:
            strategy: The strategy to implement
            workflow: The workflow execution
            activity_index: Index of the affected activity
            action: The recovery action
            
        Returns:
            True if successful, False otherwise
        """
        # Get strategy parameters
        max_retries = strategy.params.get("max_retries", 3)
        
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Check current attempt number
        current_attempt = activity.get("attempt", 0) + 1
        
        # Check if we've reached max retries
        if current_attempt > max_retries:
            logger.info(f"Max retries ({max_retries}) reached for activity {activity_index}")
            return False
            
        # Calculate backoff time
        backoff_factor = strategy.params.get("backoff_factor", 1.5)
        jitter = strategy.params.get("jitter", 0.2)
        
        backoff_time = backoff_factor ** (current_attempt - 1)
        jitter_amount = backoff_time * jitter
        backoff_time += random.uniform(-jitter_amount, jitter_amount)
        
        # Update activity for retry
        activity["attempt"] = current_attempt
        activity["status"] = ActivityStatus.PENDING
        activity["retry_info"] = {
            "attempt": current_attempt,
            "max_retries": max_retries,
            "backoff_time": backoff_time,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add timeout multiplier if specified
        timeout_multiplier = strategy.params.get("timeout_multiplier", 1.0)
        if timeout_multiplier > 1.0 and "parameters" in activity:
            if "timeout" in activity["parameters"]:
                activity["parameters"]["timeout"] *= timeout_multiplier
                
        # Log the retry
        logger.info(f"Retrying activity {activity_index} (attempt {current_attempt}/{max_retries}) after {backoff_time:.2f}s backoff")
        
        # Wait for backoff time
        await asyncio.sleep(backoff_time)
        
        # Reset the activity index in the workflow to retry this activity
        workflow.current_activity_index = activity_index - 1
        
        # Continue the workflow
        workflow_def = self.workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
        if workflow_def:
            asyncio.create_task(self.workflow_engine._continue_workflow(workflow, workflow_def))
            return True
            
        return False
        
    async def _implement_simplify_strategy(
        self,
        strategy: RecoveryStrategy,
        workflow: WorkflowExecution,
        activity_index: int,
        action: RecoveryAction
    ) -> bool:
        """
        Implement a simplification strategy.
        
        Args:
            strategy: The strategy to implement
            workflow: The workflow execution
            activity_index: Index of the affected activity
            action: The recovery action
            
        Returns:
            True if successful, False otherwise
        """
        # Get strategy parameters
        simplification_level = strategy.params.get("simplification_level", 1)
        
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Set activity status to pending for retry
        activity["status"] = ActivityStatus.PENDING
        
        # Record the simplification level
        if "parameters" not in activity:
            activity["parameters"] = {}
            
        activity["parameters"]["simplified"] = True
        activity["parameters"]["simplification_level"] = simplification_level
        
        # Apply simplification based on activity type
        activity_type = activity.get("type")
        
        if activity_type == "computation":
            # Simplify computation
            self._simplify_computation(activity, simplification_level)
            
        elif activity_type == "visualization":
            # Simplify visualization
            self._simplify_visualization(activity, simplification_level)
            
        elif activity_type == "query":
            # Simplify query
            self._simplify_query(activity, simplification_level)
            
        # Record in context that simplification was used
        if "recovery" not in workflow.context:
            workflow.context["recovery"] = {
                "simplifications": [],
                "approximations": [],
                "skips": []
            }
            
        workflow.context["recovery"]["simplifications"].append({
            "activity_index": activity_index,
            "activity_name": activity.get("name"),
            "simplification_level": simplification_level,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Log the simplification
        logger.info(f"Simplified activity {activity_index} to level {simplification_level}")
        
        # Reset the activity index in the workflow to retry this activity
        workflow.current_activity_index = activity_index - 1
        
        # Continue the workflow
        workflow_def = self.workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
        if workflow_def:
            asyncio.create_task(self.workflow_engine._continue_workflow(workflow, workflow_def))
            return True
            
        return False
        
    def _simplify_computation(self, activity: Dict[str, Any], level: int):
        """
        Simplify a computation activity.
        
        Args:
            activity: The activity to simplify
            level: Simplification level (1-3)
        """
        if "parameters" not in activity:
            return
            
        params = activity["parameters"]
        
        # Level 1: Basic simplification
        if level >= 1:
            # Disable step-by-step to reduce complexity
            params["step_by_step"] = False
            
            # Simplify expression if possible
            if "expression" in params:
                params["simplify_first"] = True
                
        # Level 2: More aggressive simplification
        if level >= 2:
            # Use numerical methods when possible
            params["numerical_fallback"] = True
            
            # Reduce precision requirements
            params["precision"] = "reduced"
            
            # Simplify domain-specific parameters
            if params.get("domain") == "calculus":
                # Favor numerical methods for calculus
                params["method"] = "numerical"
                
            elif params.get("domain") == "linear_algebra":
                # Use simpler methods for linear algebra
                params["use_decomposition"] = False
                
        # Level 3: Maximum simplification
        if level >= 3:
            # Use approximate computation
            params["approximate"] = True
            
            # Calculate only essential outputs
            params["minimal_output"] = True
            
            # Override operation if needed
            if params.get("operation") in ["integrate", "solve_differential_equation"]:
                # Switch to simpler operation
                params["operation"] = "approximate_" + params["operation"]
                
    def _simplify_visualization(self, activity: Dict[str, Any], level: int):
        """
        Simplify a visualization activity.
        
        Args:
            activity: The activity to simplify
            level: Simplification level (1-3)
        """
        if "parameters" not in activity:
            return
            
        params = activity["parameters"]
        
        # Level 1: Basic simplification
        if level >= 1:
            # Reduce resolution
            params["width"] = 600
            params["height"] = 400
            params["dpi"] = 72
            
            # Simplify styling
            params["minimal_styling"] = True
            
            # Disable animations
            params["animated"] = False
            
        # Level 2: More aggressive simplification
        if level >= 2:
            # Switch to simpler visualization type
            if params.get("visualization_type") == "function_plot_3d":
                params["visualization_type"] = "function_plot_2d"
                
            # Reduce data points
            params["max_points"] = 100
            
            # Disable interactivity
            params["interactive"] = False
            
        # Level 3: Maximum simplification
        if level >= 3:
            # Simple line plot only
            params["visualization_type"] = "simple_line"
            
            # Minimal labeling
            params["minimal_labels"] = True
            
            # Limit to a single series
            if "data" in params and isinstance(params["data"], list) and len(params["data"]) > 1:
                params["data"] = [params["data"][0]]
                
    def _simplify_query(self, activity: Dict[str, Any], level: int):
        """
        Simplify a query activity.
        
        Args:
            activity: The activity to simplify
            level: Simplification level (1-3)
        """
        if "parameters" not in activity:
            return
            
        params = activity["parameters"]
        
        # Level 1: Basic simplification
        if level >= 1:
            # Simplify response requirements
            params["simplified_response"] = True
            
            # Reduce comprehensiveness
            if "include_examples" in params:
                params["include_examples"] = False
                
            if "include_proofs" in params:
                params["include_proofs"] = False
                
        # Level 2: More aggressive simplification
        if level >= 2:
            # Focus on essential content
            params["concise"] = True
            
            # Limit detail level
            params["detail_level"] = "basic"
            
            # Disable advanced features
            if "include_steps" in params:
                params["include_steps"] = False
                
        # Level 3: Maximum simplification
        if level >= 3:
            # Minimal response only
            params["minimal_response"] = True
            
            # Plain text rather than structured format
            params["format"] = "text"
            
    async def _implement_approximate_strategy(
        self,
        strategy: RecoveryStrategy,
        workflow: WorkflowExecution,
        activity_index: int,
        action: RecoveryAction
    ) -> bool:
        """
        Implement an approximation strategy.
        
        Args:
            strategy: The strategy to implement
            workflow: The workflow execution
            activity_index: Index of the affected activity
            action: The recovery action
            
        Returns:
            True if successful, False otherwise
        """
        # Get strategy parameters
        precision = strategy.params.get("precision", "medium")
        numerical_fallback = strategy.params.get("numerical_fallback", True)
        
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Only applicable to computation activities
        if activity.get("type") != "computation":
            return False
            
        # Set activity status to pending for retry
        activity["status"] = ActivityStatus.PENDING
        
        # Update activity parameters
        if "parameters" not in activity:
            activity["parameters"] = {}
            
        params = activity["parameters"]
        
        # Set approximation parameters
        params["approximate"] = True
        params["precision"] = precision
        params["numerical_fallback"] = numerical_fallback
        
        # Adjust parameters based on precision
        if precision == "low":
            params["decimal_precision"] = 3
            params["error_tolerance"] = 1e-3
        elif precision == "medium":
            params["decimal_precision"] = 6
            params["error_tolerance"] = 1e-6
        else:  # high
            params["decimal_precision"] = 10
            params["error_tolerance"] = 1e-10
            
        # Record in context that approximation was used
        if "recovery" not in workflow.context:
            workflow.context["recovery"] = {
                "simplifications": [],
                "approximations": [],
                "skips": []
            }
            
        workflow.context["recovery"]["approximations"].append({
            "activity_index": activity_index,
            "activity_name": activity.get("name"),
            "precision": precision,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Log the approximation
        logger.info(f"Using approximation for activity {activity_index} with {precision} precision")
        
        # Reset the activity index in the workflow to retry this activity
        workflow.current_activity_index = activity_index - 1
        
        # Continue the workflow
        workflow_def = self.workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
        if workflow_def:
            asyncio.create_task(self.workflow_engine._continue_workflow(workflow, workflow_def))
            return True
            
        return False
        
    async def _implement_alternative_agent_strategy(
        self,
        strategy: RecoveryStrategy,
        workflow: WorkflowExecution,
        activity_index: int,
        action: RecoveryAction
    ) -> bool:
        """
        Implement a strategy to use an alternative agent.
        
        Args:
            strategy: The strategy to implement
            workflow: The workflow execution
            activity_index: Index of the affected activity
            action: The recovery action
            
        Returns:
            True if successful, False otherwise
        """
        # Get strategy parameters
        capability = strategy.params.get("capability")
        preferred_types = strategy.params.get("preferred_types", [])
        
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Need a capability to continue
        if not capability and not activity.get("capability"):
            return False
            
        capability = capability or activity.get("capability")
        
        # Set activity status to pending for retry
        activity["status"] = ActivityStatus.PENDING
        
        from ..agents.load_balancer import get_load_balancer
        load_balancer = get_load_balancer()
        
        # Get the current agent to exclude it
        current_agent = activity.get("agent")
        exclude_agents = [current_agent] if current_agent else []
        
        # Find an alternative agent
        alternative_agent = load_balancer.select_agent(
            capability=capability,
            consider_load=True,
            exclude_agents=exclude_agents
        )
        
        if not alternative_agent:
            logger.warning(f"No alternative agent found for capability: {capability}")
            return False
            
        # Update the activity with the new agent
        activity["agent"] = alternative_agent
        activity["original_agent"] = current_agent
        
        # Log the agent change
        logger.info(f"Switched from agent {current_agent} to {alternative_agent} for activity {activity_index}")
        
        # Reset the activity index in the workflow to retry this activity
        workflow.current_activity_index = activity_index - 1
        
        # Continue the workflow
        workflow_def = self.workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
        if workflow_def:
            asyncio.create_task(self.workflow_engine._continue_workflow(workflow, workflow_def))
            return True
            
        return False
        
    async def _implement_skip_strategy(
        self,
        strategy: RecoveryStrategy,
        workflow: WorkflowExecution,
        activity_index: int,
        action: RecoveryAction
    ) -> bool:
        """
        Implement a strategy to skip the failed operation.
        
        Args:
            strategy: The strategy to implement
            workflow: The workflow execution
            activity_index: Index of the affected activity
            action: The recovery action
            
        Returns:
            True if successful, False otherwise
        """
        # Get strategy parameters
        provide_explanation = strategy.params.get("provide_explanation", True)
        
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Mark the activity as completed with a special skip result
        skip_result = {
            "skipped": True,
            "reason": "Activity failed and was skipped as part of recovery",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add any error information
        if "error" in activity:
            skip_result["error"] = activity["error"]
            
        # Update activity status
        workflow.set_activity_status(
            activity_index=activity_index,
            status=ActivityStatus.COMPLETED,  # Mark as completed even though we're skipping
            result=skip_result
        )
        
        # Record in context that a step was skipped
        if "recovery" not in workflow.context:
            workflow.context["recovery"] = {
                "simplifications": [],
                "approximations": [],
                "skips": []
            }
            
        workflow.context["recovery"]["skips"].append({
            "activity_index": activity_index,
            "activity_name": activity.get("name"),
            "activity_type": activity.get("type"),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Add explanation to context if requested
        if provide_explanation:
            workflow.context["recovery"]["explanations"] = workflow.context["recovery"].get("explanations", [])
            workflow.context["recovery"]["explanations"].append({
                "activity_name": activity.get("name"),
                "explanation": f"The {activity.get('type', 'operation')} '{activity.get('name', 'unknown')}' "
                               f"could not be completed due to an error and was skipped.",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        # Log the skip
        logger.info(f"Skipping activity {activity_index} due to unrecoverable error")
        
        # Continue the workflow from the next activity
        workflow_def = self.workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
        if workflow_def:
            asyncio.create_task(self.workflow_engine._continue_workflow(workflow, workflow_def))
            return True
            
        return False
        
    async def _implement_minimal_response_strategy(
        self,
        strategy: RecoveryStrategy,
        workflow: WorkflowExecution,
        activity_index: int,
        action: RecoveryAction
    ) -> bool:
        """
        Implement a strategy to provide a minimal response.
        
        Args:
            strategy: The strategy to implement
            workflow: The workflow execution
            activity_index: Index of the affected activity
            action: The recovery action
            
        Returns:
            True if successful, False otherwise
        """
        # Add a special fallback response activity
        fallback_response_activity = {
            "id": str(uuid.uuid4()),
            "type": "query",
            "name": "generate_fallback_response",
            "description": "Generate minimal fallback response",
            "agent": "core_llm_agent",
            "capability": "generate_response",
            "parameters": {
                "include_steps": False,
                "include_visualization": False,
                "response_type": "fallback",
                "format": "text",
                "is_fallback": True,
                "include_error_details": True
            },
            "status": ActivityStatus.PENDING,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        # Add the fallback activity
        workflow.add_activity(fallback_response_activity)
        
        # Skip to the fallback activity
        workflow.current_activity_index = len(workflow.activities) - 2
        
        # Set workflow to running
        workflow.update_status(WorkflowExecutionStatus.RUNNING)
        
        # Continue the workflow
        workflow_def = self.workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
        if workflow_def:
            asyncio.create_task(self.workflow_engine._continue_workflow(workflow, workflow_def))
            return True
            
        return False


# Create singleton instance
_recovery_manager_instance = None

def get_recovery_manager() -> RecoveryManager:
    """Get or create the recovery manager singleton instance."""
    global _recovery_manager_instance
    if _recovery_manager_instance is None:
        _recovery_manager_instance = RecoveryManager()
    return _recovery_manager_instance


# Domain-specific recovery strategies
class MathematicalDomainRecovery:
    """
    Domain-specific recovery strategies for mathematical domains.
    
    This class provides specialized recovery strategies for different
    mathematical domains, like calculus, algebra, statistics, etc.
    """
    
    @staticmethod
    async def register_domain_handlers(manager: RecoveryManager):
        """
        Register domain-specific handlers with the recovery manager.
        
        Args:
            manager: The recovery manager
        """
        # Register handlers for computation errors
        manager.register_recovery_handler(
            ErrorCategory.COMPUTATION,
            MathematicalDomainRecovery.handle_calculus_error
        )
        
        manager.register_recovery_handler(
            ErrorCategory.COMPUTATION,
            MathematicalDomainRecovery.handle_algebra_error
        )
        
        manager.register_recovery_handler(
            ErrorCategory.COMPUTATION,
            MathematicalDomainRecovery.handle_linear_algebra_error
        )
        
        manager.register_recovery_handler(
            ErrorCategory.COMPUTATION,
            MathematicalDomainRecovery.handle_statistics_error
        )
        
    @staticmethod
    async def handle_calculus_error(
        error: ErrorClassification,
        workflow: WorkflowExecution,
        activity_index: int
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Handle errors in calculus domain.
        
        Args:
            error: The error to handle
            workflow: The workflow execution
            activity_index: Index of the affected activity
            
        Returns:
            Tuple of (success, recovery_action)
        """
        # Check if this is a calculus-related activity
        domain = workflow.context.get("domain")
        if domain != "calculus":
            return False, None
            
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Check activity type
        if activity.get("type") != "computation":
            return False, None
            
        # Get activity parameters
        params = activity.get("parameters", {})
        operation = params.get("operation")
        
        # Integration-specific handling
        if operation == "integrate":
            # Create a recovery action for using numerical integration
            action = RecoveryAction(
                action_type="numerical_integration",
                params={
                    "method": "adaptive_quad",
                    "error_tolerance": 1e-6
                },
                error=error,
                workflow_id=workflow.workflow_id,
                activity_index=activity_index
            )
            
            # Update activity parameters
            params["numerical_integration"] = True
            params["integration_method"] = "adaptive_quad"
            params["error_tolerance"] = 1e-6
            
            # Reset activity for retry
            activity["status"] = ActivityStatus.PENDING
            
            # Reset workflow to retry the activity
            workflow.current_activity_index = activity_index - 1
            
            # Continue the workflow
            workflow_engine = get_workflow_engine()
            workflow_def = workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
            if workflow_def:
                asyncio.create_task(workflow_engine._continue_workflow(workflow, workflow_def))
                return True, action
                
        # Differential equation-specific handling
        elif operation == "solve_differential_equation":
            # Create a recovery action for using numerical methods
            action = RecoveryAction(
                action_type="numerical_ode_solver",
                params={
                    "method": "runge_kutta",
                    "initial_conditions": params.get("initial_conditions", {})
                },
                error=error,
                workflow_id=workflow.workflow_id,
                activity_index=activity_index
            )
            
            # Update activity parameters
            params["numerical_solver"] = True
            params["solver_method"] = "runge_kutta"
            
            # Reset activity for retry
            activity["status"] = ActivityStatus.PENDING
            
            # Reset workflow to retry the activity
            workflow.current_activity_index = activity_index - 1
            
            # Continue the workflow
            workflow_engine = get_workflow_engine()
            workflow_def = workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
            if workflow_def:
                asyncio.create_task(workflow_engine._continue_workflow(workflow, workflow_def))
                return True, action
                
        return False, None
        
    @staticmethod
    async def handle_algebra_error(
        error: ErrorClassification,
        workflow: WorkflowExecution,
        activity_index: int
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Handle errors in algebra domain.
        
        Args:
            error: The error to handle
            workflow: The workflow execution
            activity_index: Index of the affected activity
            
        Returns:
            Tuple of (success, recovery_action)
        """
        # Implementation for algebra-specific error handling
        domain = workflow.context.get("domain")
        if domain != "algebra":
            return False, None
            
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Check activity type
        if activity.get("type") != "computation":
            return False, None
            
        # Get activity parameters
        params = activity.get("parameters", {})
        operation = params.get("operation")
        
        # Equation solving-specific handling
        if operation == "solve_equation":
            expression = params.get("expression", "")
            
            # Check if it's a high-degree polynomial
            if "^" in expression and any(f"^{i}" in expression for i in range(5, 10)):
                # Create a recovery action for using numerical root finding
                action = RecoveryAction(
                    action_type="numerical_root_finding",
                    params={
                        "method": "newton",
                        "initial_guesses": [-10, -1, 0, 1, 10]
                    },
                    error=error,
                    workflow_id=workflow.workflow_id,
                    activity_index=activity_index
                )
                
                # Update activity parameters
                params["numerical_solution"] = True
                params["root_finding_method"] = "newton"
                params["initial_guesses"] = [-10, -1, 0, 1, 10]
                
                # Reset activity for retry
                activity["status"] = ActivityStatus.PENDING
                
                # Reset workflow to retry the activity
                workflow.current_activity_index = activity_index - 1
                
                # Continue the workflow
                workflow_engine = get_workflow_engine()
                workflow_def = workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
                if workflow_def:
                    asyncio.create_task(workflow_engine._continue_workflow(workflow, workflow_def))
                    return True, action
                    
        return False, None
        
    @staticmethod
    async def handle_linear_algebra_error(
        error: ErrorClassification,
        workflow: WorkflowExecution,
        activity_index: int
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Handle errors in linear algebra domain.
        
        Args:
            error: The error to handle
            workflow: The workflow execution
            activity_index: Index of the affected activity
            
        Returns:
            Tuple of (success, recovery_action)
        """
        # Implementation for linear algebra-specific error handling
        domain = workflow.context.get("domain")
        if domain != "linear_algebra":
            return False, None
            
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Check activity type
        if activity.get("type") != "computation":
            return False, None
            
        # Get activity parameters
        params = activity.get("parameters", {})
        operation = params.get("operation")
        
        # Matrix inversion-specific handling
        if operation == "invert_matrix" or operation == "solve_system":
            # Create a recovery action for using pseudo-inverse
            action = RecoveryAction(
                action_type="pseudo_inverse",
                params={
                    "method": "svd",
                    "tolerance": 1e-10
                },
                error=error,
                workflow_id=workflow.workflow_id,
                activity_index=activity_index
            )
            
            # Update activity parameters
            params["use_pseudo_inverse"] = True
            params["pseudo_inverse_method"] = "svd"
            params["tolerance"] = 1e-10
            
            # Reset activity for retry
            activity["status"] = ActivityStatus.PENDING
            
            # Reset workflow to retry the activity
            workflow.current_activity_index = activity_index - 1
            
            # Continue the workflow
            workflow_engine = get_workflow_engine()
            workflow_def = workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
            if workflow_def:
                asyncio.create_task(workflow_engine._continue_workflow(workflow, workflow_def))
                return True, action
                
        return False, None
        
    @staticmethod
    async def handle_statistics_error(
        error: ErrorClassification,
        workflow: WorkflowExecution,
        activity_index: int
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Handle errors in statistics domain.
        
        Args:
            error: The error to handle
            workflow: The workflow execution
            activity_index: Index of the affected activity
            
        Returns:
            Tuple of (success, recovery_action)
        """
        # Implementation for statistics-specific error handling
        domain = workflow.context.get("domain")
        if domain != "statistics":
            return False, None
            
        # Get the activity
        activity = workflow.activities[activity_index]
        
        # Check activity type
        if activity.get("type") != "computation":
            return False, None
            
        # Get activity parameters
        params = activity.get("parameters", {})
        operation = params.get("operation")
        
        # Distribution fitting-specific handling
        if operation == "fit_distribution":
            # Create a recovery action for using non-parametric methods
            action = RecoveryAction(
                action_type="non_parametric_method",
                params={
                    "method": "kernel_density",
                    "bandwidth": "scott"
                },
                error=error,
                workflow_id=workflow.workflow_id,
                activity_index=activity_index
            )
            
            # Update activity parameters
            params["non_parametric"] = True
            params["density_method"] = "kernel_density"
            params["bandwidth"] = "scott"
            
            # Reset activity for retry
            activity["status"] = ActivityStatus.PENDING
            
            # Reset workflow to retry the activity
            workflow.current_activity_index = activity_index - 1
            
            # Continue the workflow
            workflow_engine = get_workflow_engine()
            workflow_def = workflow_engine.workflow_registry.get_workflow(workflow.workflow_type)
            if workflow_def:
                asyncio.create_task(workflow_engine._continue_workflow(workflow, workflow_def))
                return True, action
                
        return False, None


# Register domain-specific handlers
async def register_domain_handlers():
    """Register domain-specific handlers with the recovery manager."""
    recovery_manager = get_recovery_manager()
    await MathematicalDomainRecovery.register_domain_handlers(recovery_manager)

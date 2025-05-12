"""
Error handling and retry mechanisms for workflow orchestration.

This module provides utilities for handling errors and implementing retry
strategies during workflow execution.
"""
import logging
import time
import random
import math
from typing import Dict, Any, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)

class RetryStrategy:
    """
    Retry strategy implementation with configurable backoff.
    
    This class implements various retry strategies including:
    - Constant delay
    - Linear backoff
    - Exponential backoff
    - With or without jitter
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: Optional[float] = None,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        strategy: str = "exponential"
    ):
        """
        Initialize the retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Factor to increase delay on each retry
            jitter: Randomness factor for delay (0-1)
            strategy: Retry strategy ("constant", "linear", "exponential")
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay or 300.0  # Default: 5 minutes
        self.backoff_factor = backoff_factor
        self.jitter = min(max(0.0, jitter), 1.0)  # Clamp between 0 and 1
        
        valid_strategies = ["constant", "linear", "exponential"]
        if strategy not in valid_strategies:
            logger.warning(f"Invalid retry strategy: {strategy}, using exponential")
            strategy = "exponential"
        
        self.strategy = strategy
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a retry attempt.
        
        Args:
            attempt: The current retry attempt (1-based)
            
        Returns:
            Delay in seconds
        """
        if attempt <= 0:
            return 0.0
        
        # Calculate base delay based on strategy
        if self.strategy == "constant":
            delay = self.initial_delay
        elif self.strategy == "linear":
            delay = self.initial_delay * attempt
        else:  # exponential
            delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        
        # Apply maximum delay cap
        delay = min(delay, self.max_delay)
        
        # Apply jitter if configured
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay = delay - jitter_amount + (2 * jitter_amount * random.random())
        
        return delay
    
    def should_retry(self, attempt: int, error: Optional[Exception] = None) -> Tuple[bool, float]:
        """
        Determine if another retry should be attempted.
        
        Args:
            attempt: The current retry attempt (1-based)
            error: The error that occurred
            
        Returns:
            Tuple of (should retry, delay in seconds)
        """
        # Check if we've reached the maximum number of retries
        if attempt >= self.max_retries:
            return False, 0.0
        
        # Calculate delay for the next attempt
        delay = self.get_delay(attempt + 1)
        
        # Add specialization for certain error types
        # For example, don't retry on certain errors
        if error is not None:
            # Don't retry on validation errors
            if isinstance(error, ValueError):
                return False, 0.0
            
            # Don't retry on attribute errors
            if isinstance(error, AttributeError):
                return False, 0.0
        
        return True, delay
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RetryStrategy':
        """
        Create a retry strategy from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RetryStrategy object
        """
        return cls(
            max_retries=config.get("max_retries", 3),
            initial_delay=config.get("initial_delay", 1.0),
            max_delay=config.get("max_delay"),
            backoff_factor=config.get("backoff_factor", 2.0),
            jitter=config.get("jitter", 0.1),
            strategy=config.get("strategy", "exponential")
        )

class ErrorHandler:
    """
    Error handler for workflow orchestration.
    
    This class provides utilities for handling errors during workflow execution,
    including categorization, logging, and determining retry strategies.
    """
    
    def __init__(self, default_retry_strategy: Optional[RetryStrategy] = None):
        """
        Initialize the error handler.
        
        Args:
            default_retry_strategy: Default retry strategy
        """
        self.default_retry_strategy = default_retry_strategy or RetryStrategy()
        
        # Registry of error types to retry strategies
        self.error_strategies = {}
        
        # Register some common error types with custom strategies
        self.register_error_strategy(
            ConnectionError,
            RetryStrategy(
                max_retries=5,
                initial_delay=2.0,
                backoff_factor=2.0,
                jitter=0.2
            )
        )
        
        self.register_error_strategy(
            TimeoutError,
            RetryStrategy(
                max_retries=3,
                initial_delay=5.0,
                backoff_factor=1.5,
                jitter=0.1
            )
        )
    
    def register_error_strategy(self, error_type: type, strategy: RetryStrategy) -> None:
        """
        Register a retry strategy for a specific error type.
        
        Args:
            error_type: Error type
            strategy: Retry strategy
        """
        self.error_strategies[error_type] = strategy
        logger.debug(f"Registered retry strategy for {error_type.__name__}")
    
    def get_retry_strategy(self, error: Exception) -> RetryStrategy:
        """
        Get the appropriate retry strategy for an error.
        
        Args:
            error: The error that occurred
            
        Returns:
            Retry strategy
        """
        # Find the most specific error type match
        for error_type, strategy in self.error_strategies.items():
            if isinstance(error, error_type):
                return strategy
        
        # Fall back to default strategy
        return self.default_retry_strategy
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        attempt: int = 0
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Handle an error during workflow execution.
        
        Args:
            error: The error that occurred
            context: Context information about the error
            attempt: The current retry attempt (0-based)
            
        Returns:
            Tuple of (should retry, delay in seconds, error info)
        """
        # Get the appropriate retry strategy
        strategy = self.get_retry_strategy(error)
        
        # Determine if we should retry
        should_retry, delay = strategy.should_retry(attempt, error)
        
        # Log the error
        if should_retry:
            logger.warning(
                f"Error in {context.get('operation', 'operation')}: {error}. "
                f"Retrying in {delay:.2f}s (attempt {attempt+1}/{strategy.max_retries})"
            )
        else:
            logger.error(
                f"Error in {context.get('operation', 'operation')}: {error}. "
                f"Not retrying (attempt {attempt+1})"
            )
        
        # Create error information
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "attempt": attempt,
            "timestamp": time.time(),
            "should_retry": should_retry,
            "retry_delay": delay,
            "max_retries": strategy.max_retries
        }
        
        return should_retry, delay, error_info
    
    def retry_with_backoff(
        self,
        operation: Callable,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        context: Dict[str, Any] = None
    ) -> Any:
        """
        Execute an operation with retry and backoff.
        
        Args:
            operation: Operation to execute
            args: Positional arguments
            kwargs: Keyword arguments
            retry_strategy: Optional retry strategy override
            context: Optional context information
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        kwargs = kwargs or {}
        context = context or {"operation": operation.__name__}
        strategy = retry_strategy or self.default_retry_strategy
        
        attempt = 0
        last_error = None
        
        while True:
            try:
                # Execute the operation
                return operation(*args, **kwargs)
            except Exception as error:
                last_error = error
                
                # Handle the error
                should_retry, delay, error_info = self.handle_error(error, context, attempt)
                
                if should_retry:
                    # Sleep before retry
                    time.sleep(delay)
                    attempt += 1
                else:
                    # Raise the last error
                    raise error

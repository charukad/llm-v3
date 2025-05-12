"""
Logging configuration for the Mathematical Multimodal LLM System.

This module provides a standardized logging setup for all components.
"""
import logging
import os
import sys
import json
from typing import Dict, Any, Optional
import datetime
import threading

# Global log level - can be overridden by environment variable
LOG_LEVEL = os.environ.get("MATH_LLM_LOG_LEVEL", "INFO")

# Log format with timestamp, level, component, and message
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Configure basic logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a file handler if log file path is specified
LOG_FILE = os.environ.get("MATH_LLM_LOG_FILE")
if LOG_FILE:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(file_handler)

# Thread-local storage for correlation IDs
_thread_local = threading.local()

def set_correlation_id(correlation_id: str):
    """Set a correlation ID for the current thread."""
    _thread_local.correlation_id = correlation_id

def get_correlation_id() -> Optional[str]:
    """Get the correlation ID for the current thread."""
    return getattr(_thread_local, "correlation_id", None)


class CorrelationFilter(logging.Filter):
    """Logging filter to add correlation ID to log records."""
    
    def filter(self, record):
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = ""
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add correlation ID if available
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            log_data["correlation_id"] = correlation_id
            
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
            
        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name and standard configuration.
    
    Args:
        name: Logger name, usually the module name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Configure JSON logging if enabled
    if os.environ.get("MATH_LLM_JSON_LOGGING", "").lower() == "true":
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Add JSON handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    # Add correlation ID filter if not already added
    for log_filter in logger.filters:
        if isinstance(log_filter, CorrelationFilter):
            break
    else:
        logger.addFilter(CorrelationFilter())
        
    return logger


def log_function_call(logger: logging.Logger, level=logging.DEBUG):
    """
    Decorator to log function calls with parameters and results.
    
    Args:
        logger: Logger instance to use
        level: Logging level for the messages
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log function call
            func_args = ", ".join([repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()])
            logger.log(level, f"Calling {func.__name__}({func_args})")
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log result
                logger.log(level, f"{func.__name__} returned: {repr(result)}")
                return result
                
            except Exception as e:
                # Log exception
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
    return decorator

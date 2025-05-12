#!/usr/bin/env python3
"""
Test script for the logging system.

This script tests various logging features including JSON formatting,
correlation IDs, and function call logging.
"""
import os
import sys
import logging
import asyncio
import threading

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestration.monitoring.logger import (
    get_logger, set_correlation_id, get_correlation_id, log_function_call
)


# Create test logger
logger = get_logger("test_logging")

# Enable JSON logging for testing
os.environ["MATH_LLM_JSON_LOGGING"] = "true"


@log_function_call(logger)
def test_function(a, b, c=None):
    """Test function that will be logged."""
    logger.info(f"Inside test function with a={a}, b={b}, c={c}")
    if c is None:
        logger.warning("Parameter c is None")
    return a + b


@log_function_call(logger)
def function_with_exception():
    """Test function that raises an exception."""
    logger.info("About to raise an exception")
    raise ValueError("This is a test exception")


async def test_async_logging():
    """Test logging in asynchronous context."""
    correlation_id = "async-task-12345"
    set_correlation_id(correlation_id)
    
    logger.info(f"Async task started with correlation ID: {get_correlation_id()}")
    
    await asyncio.sleep(0.1)
    
    logger.info("Async task completed")


def test_thread_logging():
    """Test logging in a separate thread."""
    correlation_id = f"thread-{threading.get_ident()}"
    set_correlation_id(correlation_id)
    
    logger.info(f"Thread started with correlation ID: {get_correlation_id()}")
    
    # Sleep to simulate work
    import time
    time.sleep(0.1)
    
    logger.info("Thread completed")


def main():
    """Run logging tests."""
    # Test basic logging
    logger.info("Starting logging tests")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test with correlation ID
    set_correlation_id("test-correlation-123")
    logger.info(f"Using correlation ID: {get_correlation_id()}")
    
    # Test function call logging
    result = test_function(10, 20, c="test")
    logger.info(f"Function result: {result}")
    
    # Test exception logging
    try:
        function_with_exception()
    except Exception as e:
        logger.info(f"Caught exception: {str(e)}")
    
    # Test async logging
    asyncio.run(test_async_logging())
    
    # Test threaded logging
    threads = []
    for i in range(3):
        thread = threading.Thread(target=test_thread_logging)
        threads.append(thread)
        thread.start()
    
    # Wait for threads to complete
    for thread in threads:
        thread.join()
    
    logger.info("Logging tests completed")


if __name__ == "__main__":
    main()

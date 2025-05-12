"""
Test script for the message bus and MCP protocol.

This script tests the basic functionality of the message formats and handling.
"""

import logging
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_message_formats():
    """Test message format creation and serialization."""
    logger.info("Testing message formats...")
    
    try:
        from orchestration.message_bus.message_formats import (
            create_message_id,
            create_request_message,
            create_response_message,
            create_notification_message,
            create_error_message,
            serialize_message,
            deserialize_message
        )
        
        # Test message ID creation
        message_id = create_message_id()
        logger.info(f"Created message ID: {message_id}")
        
        # Test request message creation
        request = create_request_message(
            sender="test_sender",
            recipient="test_recipient",
            message_type="test_request",
            body={"test_key": "test_value"}
        )
        logger.info(f"Created request message: {json.dumps(request, indent=2)}")
        
        # Test response message creation
        response = create_response_message(
            original_message=request,
            body={"result": "success", "data": {"key": "value"}}
        )
        logger.info(f"Created response message: {json.dumps(response, indent=2)}")
        
        # Test notification message creation
        notification = create_notification_message(
            sender="test_sender",
            recipients=["recipient1", "recipient2"],
            notification_type="test_notification",
            body={"status": "active"}
        )
        logger.info(f"Created notification message: {json.dumps(notification, indent=2)}")
        
        # Test error message creation
        error = create_error_message(
            original_message=request,
            error_code="TEST_ERROR",
            error_message="This is a test error",
            error_details={"source": "test_script"}
        )
        logger.info(f"Created error message: {json.dumps(error, indent=2)}")
        
        # Test serialization and deserialization
        serialized = serialize_message(request)
        logger.info(f"Serialized message: {serialized}")
        
        deserialized = deserialize_message(serialized)
        logger.info(f"Deserialized message: {json.dumps(deserialized, indent=2)}")
        
        # Verify deserialization
        assert deserialized == request, "Deserialized message does not match original"
        
        logger.info("Message formats test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Message formats test failed: {e}")
        return False

def test_message_handler():
    """Test message handling."""
    logger.info("Testing message handler...")
    
    try:
        from orchestration.message_bus.message_formats import create_request_message
        from orchestration.message_bus.message_handler import MessageHandler
        
        # Create handler
        handler = MessageHandler()
        
        # Register test handlers
        def handle_test_request(message):
            logger.info(f"Handling test request: {message['header']['message_id']}")
            # Create a response
            return handler.create_success_response(
                original_message=message,
                response_data={"result": "Request processed successfully"}
            )
        
        def handle_compute_request(message):
            logger.info(f"Handling compute request: {message['header']['message_id']}")
            # Simulate computation
            expression = message["body"].get("expression", "")
            result = f"Computed result for {expression}"
            return handler.create_success_response(
                original_message=message,
                response_data={"result": result}
            )
        
        # Register handlers
        handler.register_handler("test_request", handle_test_request)
        handler.register_handler("compute_request", handle_compute_request)
        
        # Test message validation
        valid_message = create_request_message(
            sender="test_sender",
            recipient="test_recipient",
            message_type="test_request",
            body={"test_key": "test_value"}
        )
        
        invalid_message = {"not_a_proper_message": True}
        
        is_valid = handler.validate_message(valid_message)
        logger.info(f"Valid message validation: {is_valid}")
        
        is_invalid = not handler.validate_message(invalid_message)
        logger.info(f"Invalid message rejection: {is_invalid}")
        
        # Test message handling
        response = handler.handle_message(valid_message)
        logger.info(f"Response to valid message: {json.dumps(response, indent=2)}")
        
        error_response = handler.handle_message(invalid_message)
        logger.info(f"Response to invalid message: {json.dumps(error_response, indent=2)}")
        
        # Test specific handler
        compute_message = create_request_message(
            sender="test_sender",
            recipient="compute_agent",
            message_type="compute_request",
            body={"expression": "x^2 + 5x + 6"}
        )
        
        compute_response = handler.handle_message(compute_message)
        logger.info(f"Response to compute message: {json.dumps(compute_response, indent=2)}")
        
        # Test unknown message type
        unknown_message = create_request_message(
            sender="test_sender",
            recipient="test_recipient",
            message_type="unknown_type",
            body={"test_key": "test_value"}
        )
        
        unknown_response = handler.handle_message(unknown_message)
        logger.info(f"Response to unknown message type: {json.dumps(unknown_response, indent=2)}")
        
        logger.info("Message handler test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Message handler test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting message bus tests...")
    
    # Run tests
    formats_success = test_message_formats()
    logger.info("-----------------------------------")
    
    handler_success = test_message_handler()
    
    # Report results
    logger.info("===================================")
    logger.info("Test Results:")
    logger.info(f"Message Formats: {'PASSED' if formats_success else 'FAILED'}")
    logger.info(f"Message Handler: {'PASSED' if handler_success else 'FAILED'}")
    
    if formats_success and handler_success:
        logger.info("All message bus tests completed successfully")
        sys.exit(0)
    else:
        logger.error("One or more message bus tests failed")
        sys.exit(1)

"""
Global error handling middleware for the API.
This component is crucial for Production Readiness (Sprint 21).
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
import traceback
import logging
import time
from typing import Dict, Any, Optional, Callable
from math_llm_system.orchestration.monitoring.logger import get_logger

logger = get_logger("api.error_handler")

class ErrorHandler:
    """
    Global error handling middleware for FastAPI.
    Provides standardized error responses and logging.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.logger = logger
    
    async def __call__(self, request: Request, call_next: Callable):
        """
        Process the request and handle any exceptions.
        
        Args:
            request: The incoming request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: Either the normal response or an error response
        """
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        # Add request ID to the request state for logging
        request.state.request_id = request_id
        
        try:
            # Process the request normally
            response = await call_next(request)
            
            # Log successful requests
            process_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"Request {request_id} completed: {request.method} {request.url.path} "
                f"(Status: {response.status_code}, Time: {process_time:.2f}ms)"
            )
            
            return response
            
        except Exception as exc:
            # Log the error with traceback
            process_time = (time.time() - start_time) * 1000
            self.logger.error(
                f"Request {request_id} failed: {request.method} {request.url.path} "
                f"(Time: {process_time:.2f}ms)"
            )
            self.logger.error(f"Error details: {str(exc)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Determine appropriate status code and error message
            status_code, error_response = self._build_error_response(exc, request_id)
            
            # Return standardized error response
            return JSONResponse(
                status_code=status_code,
                content=error_response
            )
    
    def _build_error_response(self, exc: Exception, request_id: str) -> tuple[int, Dict[str, Any]]:
        """
        Build an appropriate error response based on the exception type.
        
        Args:
            exc: The exception that occurred
            request_id: The unique request identifier
            
        Returns:
            Tuple of (status_code, error_response_dict)
        """
        # Default error status and message
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_type = "internal_server_error"
        error_message = "An unexpected error occurred"
        error_details = None
        
        # Customize based on exception type
        error_name = exc.__class__.__name__
        
        # Handle different types of exceptions
        if error_name == "ValidationError":
            status_code = status.HTTP_400_BAD_REQUEST
            error_type = "validation_error"
            error_message = "Validation error in request data"
            error_details = str(exc)
            
        elif error_name == "NotFoundException":
            status_code = status.HTTP_404_NOT_FOUND
            error_type = "not_found"
            error_message = str(exc) or "Requested resource not found"
            
        elif error_name == "AuthenticationError":
            status_code = status.HTTP_401_UNAUTHORIZED
            error_type = "authentication_error"
            error_message = str(exc) or "Authentication required"
            
        elif error_name == "AuthorizationError":
            status_code = status.HTTP_403_FORBIDDEN
            error_type = "authorization_error"
            error_message = str(exc) or "Not authorized to access this resource"
            
        elif error_name == "RateLimitExceeded":
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
            error_type = "rate_limit_exceeded"
            error_message = str(exc) or "Rate limit exceeded"
            
        elif error_name == "WorkflowError":
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            error_type = "workflow_error"
            error_message = str(exc) or "Error in workflow processing"
            
        elif error_name == "DatabaseError":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            error_type = "database_error"
            error_message = "Database service unavailable"
            
        elif error_name == "MessageBusError":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            error_type = "message_bus_error"
            error_message = "Message bus service unavailable"
            
        # In development, we might want to include the actual exception details
        # In production, we would want to be more careful about what we expose
        if error_details is None and status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            # For 500 errors, don't expose internal details in production
            pass  # error_details remains None
        elif error_details is None:
            # For other errors, we can be more informative
            error_details = str(exc)
        
        # Build the standardized error response
        error_response = {
            "error": {
                "type": error_type,
                "message": error_message,
                "request_id": request_id
            }
        }
        
        # Include details if available and appropriate
        if error_details:
            error_response["error"]["details"] = error_details
        
        return status_code, error_response

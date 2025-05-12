"""
Request batching middleware for optimizing API performance.
Aggregates similar requests to reduce processing overhead and improve throughput.
"""
import time
import threading
import asyncio
import uuid
import json
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple, Set
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from math_llm_system.orchestration.monitoring.logger import get_logger

logger = get_logger("api.request_batcher")

class RequestBatcher:
    """
    Middleware for batching similar API requests to improve performance.
    Aggregates requests to reduce computational overhead for frequently performed operations.
    """
    
    def __init__(self, batch_window: float = 0.1, max_batch_size: int = 50):
        """
        Initialize request batcher middleware.
        
        Args:
            batch_window: Time window for batching requests (seconds)
            max_batch_size: Maximum number of requests in a batch
        """
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        
        # Dictionary to store pending batches by endpoint
        self.batches = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Set of batchable endpoints
        self.batchable_endpoints = {
            "/api/math/compute",
            "/api/math/expression/evaluate",
            "/api/math/expression/simplify",
            "/api/latex/render",
            "/api/visualization/plot2d"
        }
        
        # Mapping of endpoint handlers
        self.batch_handlers = {}
        
        logger.info(f"Request batcher initialized with window={batch_window}s, "
                   f"max_size={max_batch_size}")
    
    def register_batch_handler(self, endpoint: str, handler: Callable):
        """
        Register a batch handler function for an endpoint.
        
        Args:
            endpoint: API endpoint path
            handler: Batch handling function
        """
        self.batch_handlers[endpoint] = handler
        self.batchable_endpoints.add(endpoint)
        logger.info(f"Registered batch handler for endpoint: {endpoint}")
    
    async def __call__(self, request: Request, call_next: Callable):
        """
        Process the request and apply batching if appropriate.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or endpoint handler
            
        Returns:
            Response
        """
        # Get the request path
        path = request.url.path
        
        # Check if this is a batchable endpoint
        if path in self.batchable_endpoints and request.method == "POST":
            # Try to batch this request
            return await self._handle_batchable_request(request, path)
        
        # For non-batchable requests, just pass through
        return await call_next(request)
    
    async def _handle_batchable_request(self, request: Request, path: str) -> Response:
        """
        Handle a request that can be batched.
        
        Args:
            request: FastAPI request
            path: Request path
            
        Returns:
            Response
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Error parsing request body: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request body"}
            )
        
        # Create a future to get the result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Add request to batch
        with self.lock:
            if path not in self.batches:
                # Create new batch for this endpoint
                self.batches[path] = {
                    "requests": [],
                    "futures": {},
                    "created_at": time.time(),
                    "processing": False
                }
                
                # Schedule batch processing
                asyncio.create_task(self._process_batch_after_window(path))
            
            # Add to batch
            batch = self.batches[path]
            batch["requests"].append({
                "id": request_id,
                "body": body,
                "headers": dict(request.headers)
            })
            batch["futures"][request_id] = future
            
            # If batch is full, process it immediately
            if len(batch["requests"]) >= self.max_batch_size and not batch["processing"]:
                batch["processing"] = True
                asyncio.create_task(self._process_batch(path))
        
        # Wait for result
        try:
            result = await future
            
            # If result is an exception, raise it
            if isinstance(result, Exception):
                raise result
                
            return JSONResponse(
                status_code=200,
                content=result
            )
        except Exception as e:
            logger.error(f"Error processing batched request {request_id}: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing request: {str(e)}"}
            )
    
    async def _process_batch_after_window(self, path: str):
        """
        Schedule batch processing after the window expires.
        
        Args:
            path: Request path
        """
        await asyncio.sleep(self.batch_window)
        
        with self.lock:
            batch = self.batches.get(path)
            if batch and not batch["processing"] and batch["requests"]:
                batch["processing"] = True
                asyncio.create_task(self._process_batch(path))
    
    async def _process_batch(self, path: str):
        """
        Process a batch of requests.
        
        Args:
            path: Request path
        """
        # Get the batch
        with self.lock:
            batch = self.batches.get(path)
            if not batch or not batch["requests"]:
                # No requests to process
                if batch:
                    batch["processing"] = False
                return
            
            # Extract the batch data
            requests = batch["requests"]
            futures = batch["futures"]
            
            # Clear the batch
            self.batches[path] = {
                "requests": [],
                "futures": {},
                "created_at": time.time(),
                "processing": False
            }
        
        try:
            # Get the batch handler
            handler = self.batch_handlers.get(path)
            
            if handler:
                # Use registered handler
                results = await handler(requests)
            else:
                # Default batch processing
                results = await self._default_batch_processor(path, requests)
            
            # Set results for futures
            for request_id, result in results.items():
                if request_id in futures:
                    futures[request_id].set_result(result)
                else:
                    logger.warning(f"Request ID {request_id} not found in futures")
            
            # Set exceptions for any missing results
            for request_id, future in futures.items():
                if not future.done():
                    future.set_exception(Exception("Request not processed in batch"))
            
            logger.debug(f"Processed batch of {len(requests)} requests for {path}")
            
        except Exception as e:
            logger.error(f"Error processing batch for {path}: {e}")
            
            # Set exception for all futures
            for future in futures.values():
                if not future.done():
                    future.set_exception(e)
    
    async def _default_batch_processor(self, path: str, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Default batch processor for endpoints without a registered handler.
        
        Args:
            path: Request path
            requests: List of batched requests
            
        Returns:
            Dictionary mapping request IDs to results
        """
        results = {}
        
        # Process each request individually (no batching benefit, but maintains interface)
        for request in requests:
            request_id = request["id"]
            
            # Simplified approximation of endpoint behavior
            try:
                if path == "/api/math/compute":
                    results[request_id] = await self._process_math_compute(request["body"])
                elif path == "/api/math/expression/evaluate":
                    results[request_id] = await self._process_expression_evaluate(request["body"])
                elif path == "/api/math/expression/simplify":
                    results[request_id] = await self._process_expression_simplify(request["body"])
                elif path == "/api/latex/render":
                    results[request_id] = await self._process_latex_render(request["body"])
                elif path == "/api/visualization/plot2d":
                    results[request_id] = await self._process_visualization_plot2d(request["body"])
                else:
                    results[request_id] = {"error": f"No handler for endpoint: {path}"}
            except Exception as e:
                logger.error(f"Error processing request {request_id} for {path}: {e}")
                results[request_id] = {"error": str(e)}
        
        return results
    
    async def _process_math_compute(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a math computation request.
        This is a placeholder for the actual implementation.
        
        Args:
            body: Request body
            
        Returns:
            Computation result
        """
        # In a real implementation, this would call the actual handler
        return {"result": "Math computation not implemented in batching middleware"}
    
    async def _process_expression_evaluate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an expression evaluation request.
        This is a placeholder for the actual implementation.
        
        Args:
            body: Request body
            
        Returns:
            Evaluation result
        """
        # In a real implementation, this would call the actual handler
        return {"result": "Expression evaluation not implemented in batching middleware"}
    
    async def _process_expression_simplify(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an expression simplification request.
        This is a placeholder for the actual implementation.
        
        Args:
            body: Request body
            
        Returns:
            Simplification result
        """
        # In a real implementation, this would call the actual handler
        return {"result": "Expression simplification not implemented in batching middleware"}
    
    async def _process_latex_render(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a LaTeX rendering request.
        This is a placeholder for the actual implementation.
        
        Args:
            body: Request body
            
        Returns:
            Rendering result
        """
        # In a real implementation, this would call the actual handler
        return {"result": "LaTeX rendering not implemented in batching middleware"}
    
    async def _process_visualization_plot2d(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a 2D plotting request.
        This is a placeholder for the actual implementation.
        
        Args:
            body: Request body
            
        Returns:
            Plotting result
        """
        # In a real implementation, this would call the actual handler
        return {"result": "2D plotting not implemented in batching middleware"}

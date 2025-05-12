"""
Parallel processing system for mathematical computations.
Enables efficient execution of independent operations across multiple cores.
"""
import time
import threading
import os
import queue
import logging
import concurrent.futures
from typing import Dict, Any, List, Callable, Optional, Tuple, Union, TypeVar, Generic
import sympy as sp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from math_llm_system.orchestration.performance.resource_manager import ResourceManager, resource_managed
from math_llm_system.orchestration.monitoring.logger import get_logger
from math_llm_system.math_processing.computation.computation_cache import ComputationCache, cached_computation

logger = get_logger("math_processing.parallel_processor")

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

class ParallelMathProcessor:
    """
    Parallel processing system for mathematical operations.
    Enables efficient execution of multiple independent operations.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize parallel math processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
        """
        # Get resource manager for optimal resource allocation
        self.resource_manager = ResourceManager()
        
        # Determine number of workers based on available resources
        if max_workers is None:
            cpu_count = os.cpu_count()
            resource_usage = self.resource_manager.get_resource_usage()
            available_workers = resource_usage["cpu"]["available_workers"]
            
            # Use at most 75% of available workers or at least 2
            max_workers = max(2, min(available_workers, int(cpu_count * 0.75)))
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        # Create appropriate executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize cache for results
        self.cache = ComputationCache()
        
        logger.info(f"Initialized parallel math processor with {max_workers} workers "
                   f"using {'processes' if use_processes else 'threads'}")
    
    def map(self, func: Callable[[T], R], items: List[T], 
           chunk_size: Optional[int] = None) -> List[R]:
        """
        Apply function to each item in parallel.
        
        Args:
            func: Function to apply
            items: List of items to process
            chunk_size: Size of chunks for batched processing
            
        Returns:
            List of results
        """
        # No items, return empty list
        if not items:
            return []
        
        # Single item, process directly
        if len(items) == 1:
            return [func(items[0])]
        
        # Determine chunk size if not specified
        if chunk_size is None:
            # Aim for at least 2 items per worker, but no more than 20
            chunk_size = max(2, min(20, len(items) // self.max_workers))
        
        # Use executor to process items in parallel
        results = list(self.executor.map(func, items, chunksize=chunk_size))
        
        return results
    
    def process_batch(self, batch: List[Dict[str, Any]], 
                     operation: Callable[[Dict[str, Any]], Dict[str, Any]],
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of operations in parallel.
        
        Args:
            batch: List of operation parameters
            operation: Function to apply to each set of parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of operation results
        """
        # Track total and completed items for progress reporting
        total_items = len(batch)
        completed_items = 0
        
        # Use a queue to collect results in order of completion
        results_queue = queue.Queue()
        
        # Submit all operations to executor
        futures = []
        for item in batch:
            future = self.executor.submit(operation, item)
            future.operation_id = item.get("id", str(id(item)))
            
            # Add callback to track progress
            def on_complete(future):
                nonlocal completed_items
                completed_items += 1
                if progress_callback:
                    progress_callback(completed_items, total_items)
                
                # Add result to queue
                results_queue.put((future.operation_id, future))
            
            future.add_done_callback(on_complete)
            futures.append(future)
        
        # Collect results in submission order
        ordered_results = []
        operation_id_to_index = {item.get("id", str(id(item))): i for i, item in enumerate(batch)}
        
        # Wait for all operations to complete
        for future in concurrent.futures.as_completed(futures):
            operation_id = future.operation_id
            try:
                result = future.result()
                ordered_results.append((operation_id_to_index[operation_id], result))
            except Exception as e:
                logger.error(f"Error in operation {operation_id}: {e}")
                ordered_results.append((operation_id_to_index[operation_id], {
                    "success": False,
                    "error": str(e)
                }))
        
        # Sort results by original order
        ordered_results.sort(key=lambda x: x[0])
        
        return [r[1] for r in ordered_results]
    
    def parallel_symbolic_operation(self, operation_name: str, expressions: List[sp.Expr], 
                                  *args, **kwargs) -> List[sp.Expr]:
        """
        Apply a symbolic operation to multiple expressions in parallel.
        
        Args:
            operation_name: Name of the symbolic operation
            expressions: List of expressions to process
            *args: Additional positional arguments for the operation
            **kwargs: Additional keyword arguments for the operation
            
        Returns:
            List of results
        """
        # Define wrapper for the symbolic operation
        def process_expression(expr):
            try:
                # Get the operation method from sympy
                operation = getattr(sp, operation_name, None)
                if operation is None:
                    raise ValueError(f"Invalid operation: {operation_name}")
                
                # Apply operation
                result = operation(expr, *args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in symbolic operation {operation_name}: {e}")
                # Return the original expression on error
                return expr
        
        # Process expressions in parallel
        results = self.map(process_expression, expressions)
        
        return results
    
    @cached_computation(ttl=3600)
    def parallel_evaluate(self, expr: Union[sp.Expr, List[sp.Expr]], 
                        var_values: Dict[str, Union[float, int]]) -> Union[float, List[float]]:
        """
        Evaluate expression(s) with variable values in parallel.
        
        Args:
            expr: Expression or list of expressions to evaluate
            var_values: Dictionary mapping variable names to values
            
        Returns:
            Evaluated result(s)
        """
        # Handle single expression case
        if not isinstance(expr, list):
            # Create lambdified function
            var_symbols = list(var_values.keys())
            symbols = [sp.Symbol(v) for v in var_symbols]
            func = sp.lambdify(symbols, expr)
            
            # Evaluate
            values = [var_values[v] for v in var_symbols]
            try:
                return float(func(*values))
            except Exception as e:
                logger.error(f"Error evaluating expression {expr}: {e}")
                return float('nan')
        
        # Handle multiple expressions
        var_symbols = list(var_values.keys())
        symbols = [sp.Symbol(v) for v in var_symbols]
        values = [var_values[v] for v in var_symbols]
        
        # Define evaluation function
        def evaluate_single(expression):
            try:
                func = sp.lambdify(symbols, expression)
                return float(func(*values))
            except Exception as e:
                logger.error(f"Error evaluating expression {expression}: {e}")
                return float('nan')
        
        # Evaluate in parallel
        return self.map(evaluate_single, expr)
    
    def parallel_solve(self, equations: List[sp.Eq], 
                     variables: List[sp.Symbol]) -> List[Dict[sp.Symbol, sp.Expr]]:
        """
        Solve multiple equations or systems in parallel.
        
        Args:
            equations: List of equations or lists of equations
            variables: List of variables or lists of variables
            
        Returns:
            List of solution dictionaries
        """
        # Define solving function
        def solve_system(system_data):
            eqs = system_data["equations"]
            vars = system_data["variables"]
            
            try:
                # Solve the system
                solutions = sp.solve(eqs, vars, dict=True)
                return solutions
            except Exception as e:
                logger.error(f"Error solving equations {eqs}: {e}")
                return []
        
        # Prepare systems data
        systems = []
        for i, eqs in enumerate(equations):
            # Determine variables for this system
            if isinstance(variables[0], list):
                # Each system has its own variable list
                vars = variables[i] if i < len(variables) else variables[-1]
            else:
                # Same variables for all systems
                vars = variables
                
            systems.append({
                "equations": eqs,
                "variables": vars
            })
        
        # Solve systems in parallel
        return self.map(solve_system, systems)
    
    def parallel_integrate(self, expressions: List[sp.Expr], 
                         variables: List[sp.Symbol],
                         limits: Optional[List[Tuple]] = None) -> List[sp.Expr]:
        """
        Integrate multiple expressions in parallel.
        
        Args:
            expressions: List of expressions to integrate
            variables: List of variables of integration
            limits: Optional integration limits
            
        Returns:
            List of integration results
        """
        # Define integration function
        def integrate_expr(integration_data):
            expr = integration_data["expression"]
            var = integration_data["variable"]
            
            try:
                # Handle definite integral
                if "limits" in integration_data and integration_data["limits"] is not None:
                    lower, upper = integration_data["limits"]
                    result = sp.integrate(expr, (var, lower, upper))
                else:
                    # Indefinite integral
                    result = sp.integrate(expr, var)
                    
                return result
            except Exception as e:
                logger.error(f"Error integrating expression {expr}: {e}")
                return sp.S.NaN
        
        # Prepare integration data
        integration_data = []
        for i, expr in enumerate(expressions):
            # Determine variable for this expression
            var = variables[i] if i < len(variables) else variables[-1]
            
            data = {
                "expression": expr,
                "variable": var
            }
            
            # Add limits if provided
            if limits and i < len(limits):
                data["limits"] = limits[i]
            
            integration_data.append(data)
        
        # Integrate in parallel
        return self.map(integrate_expr, integration_data)
    
    def parallel_differentiate(self, expressions: List[sp.Expr], 
                             variables: List[sp.Symbol],
                             orders: Optional[List[int]] = None) -> List[sp.Expr]:
        """
        Differentiate multiple expressions in parallel.
        
        Args:
            expressions: List of expressions to differentiate
            variables: List of variables of differentiation
            orders: Optional differentiation orders
            
        Returns:
            List of differentiation results
        """
        # Define differentiation function
        def differentiate_expr(diff_data):
            expr = diff_data["expression"]
            var = diff_data["variable"]
            
            try:
                # Apply the differentiation
                if "order" in diff_data and diff_data["order"] is not None:
                    result = sp.diff(expr, var, diff_data["order"])
                else:
                    result = sp.diff(expr, var)
                    
                return result
            except Exception as e:
                logger.error(f"Error differentiating expression {expr}: {e}")
                return sp.S.NaN
        
        # Prepare differentiation data
        diff_data = []
        for i, expr in enumerate(expressions):
            # Determine variable for this expression
            var = variables[i] if i < len(variables) else variables[-1]
            
            data = {
                "expression": expr,
                "variable": var
            }
            
            # Add order if provided
            if orders and i < len(orders):
                data["order"] = orders[i]
            
            diff_data.append(data)
        
        # Differentiate in parallel
        return self.map(differentiate_expr, diff_data)
    
    def batch_compute(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute a batch of arbitrary mathematical operations in parallel.
        
        Args:
            operations: List of operation specifications
            
        Returns:
            List of operation results
        """
        # Define operation processor
        def process_operation(operation):
            op_type = operation.get("type")
            expr = operation.get("expression")
            
            # Convert expression from string if necessary
            if isinstance(expr, str):
                try:
                    expr = sp.sympify(expr)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid expression: {e}",
                        "operation": op_type
                    }
            
            try:
                # Process based on operation type
                if op_type == "simplify":
                    result = sp.simplify(expr)
                elif op_type == "expand":
                    result = sp.expand(expr)
                elif op_type == "factor":
                    result = sp.factor(expr)
                elif op_type == "solve":
                    var = operation.get("variable")
                    if isinstance(var, str):
                        var = sp.Symbol(var)
                    result = sp.solve(expr, var, dict=True)
                elif op_type == "diff":
                    var = operation.get("variable")
                    if isinstance(var, str):
                        var = sp.Symbol(var)
                    order = operation.get("order", 1)
                    result = sp.diff(expr, var, order)
                elif op_type == "integrate":
                    var = operation.get("variable")
                    if isinstance(var, str):
                        var = sp.Symbol(var)
                    limits = operation.get("limits")
                    if limits:
                        result = sp.integrate(expr, (var, limits[0], limits[1]))
                    else:
                        result = sp.integrate(expr, var)
                elif op_type == "evaluate":
                    var_values = operation.get("values", {})
                    for var_name, value in var_values.items():
                        expr = expr.subs(sp.Symbol(var_name), value)
                    result = float(expr.evalf())
                else:
                    return {
                        "success": False,
                        "error": f"Unknown operation type: {op_type}",
                        "operation": op_type
                    }
                
                return {
                    "success": True,
                    "result": result,
                    "operation": op_type,
                    "latex": sp.latex(result) if op_type != "solve" else str(result)
                }
                
            except Exception as e:
                logger.error(f"Error in {op_type} operation: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "operation": op_type
                }
        
        # Process operations in parallel
        return self.process_batch(operations, process_operation)
    
    def shutdown(self):
        """Shut down the parallel processor."""
        self.executor.shutdown()
        logger.info("Parallel math processor shut down")


# Decorator for parallel processing of multiple inputs
def parallel_compute(use_processes: bool = False, chunk_size: Optional[int] = None):
    """
    Decorator for parallel computation of multiple inputs.
    
    Args:
        use_processes: Whether to use processes instead of threads
        chunk_size: Size of chunks for batched processing
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(inputs, *args, **kwargs):
            # Initialize processor
            processor = ParallelMathProcessor(use_processes=use_processes)
            
            # Define function to apply to each input
            def process_single(input_item):
                return func(input_item, *args, **kwargs)
            
            # Process inputs in parallel
            results = processor.map(process_single, inputs, chunk_size=chunk_size)
            
            # Shutdown processor
            processor.shutdown()
            
            return results
        
        return wrapper
    
    return decorator

#!/usr/bin/env python
"""
Test script for performance optimization components of the Mathematical Multimodal LLM System.

This script demonstrates how to use the performance optimization components in a workflow.
"""

import time
import random
import argparse
import logging
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.performance.performance_optimizer import (
    get_performance_optimizer,
    OptimizationLevel,
    optimize_function
)
from orchestration.performance.metrics_visualizer import (
    get_metrics_visualizer,
    generate_current_performance_report
)
from orchestration.performance.integration import (
    optimize_orchestration_manager,
    optimize_mathematical_computation,
    optimized_computation_example
)
from orchestration.monitoring.logger import get_logger

logger = get_logger(__name__)


def simulate_mathematical_computation(complexity="medium"):
    """
    Simulate a mathematical computation.
    
    Args:
        complexity: Complexity level of the computation (low, medium, high)
    
    Returns:
        Computation result
    """
    # Simulate computation based on complexity
    if complexity == "low":
        time.sleep(0.1)
        return {"result": random.random(), "precision": "exact"}
    
    elif complexity == "medium":
        time.sleep(0.3)
        return {"result": random.random() * 10, "precision": "exact"}
    
    elif complexity == "high":
        time.sleep(0.7)
        return {"result": random.random() * 100, "precision": "approximate"}
    
    else:
        raise ValueError(f"Unknown complexity level: {complexity}")


@optimize_function(OptimizationLevel.BASIC)
def optimized_symbolic_computation(inputs):
    """
    Perform an optimized symbolic computation.
    
    Args:
        inputs: Input parameters
        
    Returns:
        Computation result
    """
    expression = inputs.get("expression", "x^2")
    logger.info(f"Performing symbolic computation on: {expression}")
    
    # Simulate work
    time.sleep(0.2)
    
    # Simulate result
    return {
        "result": f"Computed {expression}",
        "steps": [
            f"Step 1: Parse {expression}",
            "Step 2: Apply symbolic rules",
            "Step 3: Simplify result"
        ]
    }


def simulate_workflow_execution():
    """
    Simulate a workflow execution using performance optimization.
    
    Returns:
        Workflow execution statistics
    """
    logger.info("Simulating workflow execution with performance optimization")
    
    # Get the performance optimizer
    optimizer = get_performance_optimizer()
    
    # Configure optimization level
    optimizer.optimization_level = OptimizationLevel.BASIC
    
    # Simulate a series of computations
    start_time = time.time()
    
    # Perform multiple computations with caching
    results = []
    for i in range(10):
        if i < 5:
            # First 5 are unique
            expression = f"x^{i+2} + {i+1}*x"
        else:
            # Second 5 duplicate the first 5
            expression = f"x^{(i-5)+2} + {(i-5)+1}*x"
        
        inputs = {"expression": expression}
        
        result = optimizer.optimize_computation(
            optimized_symbolic_computation,
            inputs
        )
        
        results.append(result)
    
    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    return {
        "execution_time": execution_time,
        "cache_hit_ratio": metrics["cache_hit_ratio"],
        "cache_hits": metrics["cache_hits"],
        "cache_misses": metrics["cache_misses"],
        "results": results
    }


def main():
    """Main function to run performance optimization tests."""
    parser = argparse.ArgumentParser(description="Test performance optimization components")
    parser.add_argument("--report", action="store_true", help="Generate a performance report")
    parser.add_argument("--level", choices=["none", "basic", "aggressive", "adaptive"], 
                       default="basic", help="Optimization level")
    
    args = parser.parse_args()
    
    # Set optimization level
    optimizer = get_performance_optimizer()
    level_map = {
        "none": OptimizationLevel.NONE,
        "basic": OptimizationLevel.BASIC,
        "aggressive": OptimizationLevel.AGGRESSIVE,
        "adaptive": OptimizationLevel.ADAPTIVE
    }
    optimizer.optimization_level = level_map[args.level]
    
    # Run sample computations
    logger.info(f"Running performance tests with optimization level: {args.level}")
    
    # Run an optimized computation
    inputs = {"value": 42}
    result = optimized_computation_example(inputs)
    logger.info(f"Optimized computation result: {result}")
    
    # Simulate a workflow execution
    workflow_stats = simulate_workflow_execution()
    logger.info(f"Workflow execution time: {workflow_stats['execution_time']:.2f} seconds")
    logger.info(f"Cache hit ratio: {workflow_stats['cache_hit_ratio']:.2%}")
    
    # Record metrics snapshot
    metrics_visualizer = get_metrics_visualizer()
    metrics_visualizer.record_metrics_snapshot()
    
    # Generate performance report if requested
    if args.report:
        report_path = generate_current_performance_report()
        logger.info(f"Generated performance report: {report_path}")
        
        # Generate charts
        system_chart = metrics_visualizer.generate_system_load_chart()
        cache_chart = metrics_visualizer.generate_cache_performance_chart()
        computation_chart = metrics_visualizer.generate_computation_time_chart()
        
        logger.info(f"Generated system load chart: {system_chart}")
        logger.info(f"Generated cache performance chart: {cache_chart}")
        logger.info(f"Generated computation time chart: {computation_chart}")


if __name__ == "__main__":
    main()

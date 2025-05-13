#!/usr/bin/env python3
"""
Test script for all visualization types using fixed functions
"""

import os
import numpy as np
from visualization.plotting.plot_2d import plot_function_2d
from visualization.plotting.plot_3d import plot_function_3d, plot_parametric_3d
from visualization.plotting.statistical import plot_histogram, plot_scatter
from fixed_plot_test import fixed_plot_multiple_functions_2d

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

def run_visualization_tests():
    """Run tests for all visualization types"""
    
    print("Running tests for all visualization types...\n")
    results = {}
    
    # Test 1: 2D Function Plot
    print("Test 1: 2D Function Plot")
    result = plot_function_2d(
        function_expr="sin(x)",
        x_range=(-5, 5),
        title="Sine Function",
        save_path="visualizations/test_all_sine_2d.png"
    )
    results["2d_function"] = result["success"]
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    print()
    
    # Test 2: Multiple 2D Functions Plot
    print("Test 2: Multiple 2D Functions Plot")
    result = fixed_plot_multiple_functions_2d(
        functions=["sin(x)", "cos(x)", "x**2/5"],
        labels=["sin(x)", "cos(x)", "x²/5"],
        x_range=(-5, 5),
        title="Multiple Functions",
        save_path="visualizations/test_all_multiple_2d.png"
    )
    results["multiple_2d"] = result["success"]
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    print()
    
    # Test 3: 3D Function Plot
    print("Test 3: 3D Function Plot")
    result = plot_function_3d(
        function_expr="sin(sqrt(x**2 + y**2))",
        x_range=(-5, 5),
        y_range=(-5, 5),
        title="Sinc Function",
        save_path="visualizations/test_all_3d.png"
    )
    results["3d_function"] = result["success"]
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    print()
    
    # Test 4: Parametric 3D Plot
    print("Test 4: Parametric 3D Plot")
    result = plot_parametric_3d(
        x_expr="cos(t)",
        y_expr="sin(t)",
        z_expr="t/3",
        t_range=(0, 6*np.pi),
        title="3D Helix",
        save_path="visualizations/test_all_parametric_3d.png"
    )
    results["parametric_3d"] = result["success"]
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    print()
    
    # Test 5: Histogram
    print("Test 5: Histogram")
    data = np.random.normal(0, 1, 1000)
    result = plot_histogram(
        data=data,
        bins=30,
        title="Normal Distribution",
        save_path="visualizations/test_all_histogram.png"
    )
    results["histogram"] = result["success"]
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    print()
    
    # Test 6: Scatter Plot
    print("Test 6: Scatter Plot")
    x_data = np.random.uniform(-5, 5, 100)
    y_data = x_data * 2 + np.random.normal(0, 1, 100)
    result = plot_scatter(
        x_data=x_data,
        y_data=y_data,
        title="Linear Relationship with Noise",
        show_regression=True,
        save_path="visualizations/test_all_scatter.png"
    )
    results["scatter"] = result["success"]
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"File Path: {result['file_path']}")
    print()
    
    # Summary
    print("Test Summary:")
    for test_name, success in results.items():
        print(f"  {test_name}: {'PASS' if success else 'FAIL'}")
    
    all_passed = all(results.values())
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"Output Files Directory: {os.path.abspath('visualizations')}")

if __name__ == "__main__":
    run_visualization_tests() 
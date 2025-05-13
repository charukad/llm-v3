#!/usr/bin/env python3
"""
Test script for the visualization plotting capabilities
"""

import os
import numpy as np
from visualization.plotting.plot_2d import plot_function_2d, plot_multiple_functions_2d
from visualization.plotting.plot_3d import plot_function_3d, plot_parametric_3d
from visualization.plotting.statistical import plot_histogram, plot_scatter

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

# Test 2D function plotting
def test_2d_function():
    result = plot_function_2d(
        function_expr="sin(x)",
        x_range=(-5, 5),
        title="Sine Function",
        save_path="visualizations/test_sine_2d.png"
    )
    print(f"2D Function Plot: {result['success']}")
    if result['success']:
        print(f"Saved to: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

# Test multiple 2D functions
def test_multiple_2d_functions():
    result = plot_multiple_functions_2d(
        functions=["sin(x)", "cos(x)", "x^2/5"],
        labels=["sin(x)", "cos(x)", "x²/5"],
        x_range=(-5, 5),
        title="Multiple Functions",
        save_path="visualizations/test_multiple_2d.png"
    )
    print(f"Multiple 2D Functions Plot: {result['success']}")
    if result['success']:
        print(f"Saved to: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

# Test 3D function
def test_3d_function():
    result = plot_function_3d(
        function_expr="sin(sqrt(x^2 + y^2))",
        x_range=(-5, 5),
        y_range=(-5, 5),
        title="3D Sinc Function",
        save_path="visualizations/test_3d.png"
    )
    print(f"3D Function Plot: {result['success']}")
    if result['success']:
        print(f"Saved to: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

# Test 3D parametric curve
def test_3d_parametric():
    result = plot_parametric_3d(
        x_expr="cos(t)",
        y_expr="sin(t)",
        z_expr="t/3",
        t_range=(0, 6*np.pi),
        title="3D Helix",
        save_path="visualizations/test_parametric_3d.png"
    )
    print(f"3D Parametric Plot: {result['success']}")
    if result['success']:
        print(f"Saved to: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

# Test histogram
def test_histogram():
    # Generate random data
    data = np.random.normal(0, 1, 1000)
    
    result = plot_histogram(
        data=data,
        bins=30,
        title="Normal Distribution",
        save_path="visualizations/test_histogram.png"
    )
    print(f"Histogram Plot: {result['success']}")
    if result['success']:
        print(f"Saved to: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

# Test scatter plot
def test_scatter():
    # Generate correlated data
    x_data = np.random.uniform(-5, 5, 100)
    y_data = x_data * 2 + np.random.normal(0, 1, 100)
    
    result = plot_scatter(
        x_data=x_data,
        y_data=y_data,
        title="Scatter with Regression",
        show_regression=True,
        save_path="visualizations/test_scatter.png"
    )
    print(f"Scatter Plot: {result['success']}")
    if result['success']:
        print(f"Saved to: {result['file_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    print("Testing visualization functions...")
    
    # Run all tests
    test_2d_function()
    test_multiple_2d_functions()
    test_3d_function()
    test_3d_parametric()
    test_histogram()
    test_scatter()
    
    print("All tests completed. Check the 'visualizations' directory for output files.") 
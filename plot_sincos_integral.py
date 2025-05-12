#!/usr/bin/env python3
"""
Script to plot sin(x)*cos(x) and its integral.
This creates a mathematical visualization showing both the function and its antiderivative.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np

def create_sincos_integral_plot(x_min=-5, x_max=5):
    """
    Create a plot showing sin(x)*cos(x) and its integral.
    
    Args:
        x_min: Minimum x value
        x_max: Maximum x value
    
    Returns:
        Path to the saved plot file
    """
    # Generate x values
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate sin(x)*cos(x) = 0.5*sin(2x)
    y_function = np.sin(x) * np.cos(x)
    
    # The integral of sin(x)*cos(x) = -0.5*cos(2x) + C
    # For simplicity, we'll set C = 0
    y_integral = -0.5 * np.cos(2*x)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the function and its integral
    plt.plot(x, y_function, 'b-', label='sin(x)·cos(x)')
    plt.plot(x, y_integral, 'r-', label='∫sin(x)·cos(x)dx = -0.5·cos(2x)')
    
    # Add labels and title
    plt.title('Plot of sin(x)·cos(x) and its Integral')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.legend()
    
    # Save the plot to a file
    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/sincos_integral_{int(time.time())}.png"
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

if __name__ == "__main__":
    plot_file = create_sincos_integral_plot()
    print(f"Plot saved to: {os.path.abspath(plot_file)}") 
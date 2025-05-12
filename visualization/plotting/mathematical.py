"""
Specialized mathematical visualizations.

This module provides visualizations specifically designed for common
mathematical concepts such as complex numbers, linear algebra, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Union, Tuple
import io
import base64
import os
import uuid
from datetime import datetime

def plot_complex_numbers(
    complex_numbers: List[complex],
    labels: Optional[List[str]] = None,
    title: str = "Complex Numbers",
    show_unit_circle: bool = True,
    show_real_axis: bool = True,
    show_imag_axis: bool = True,
    show_abs_value: bool = False,
    show_phase: bool = False,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Visualize complex numbers in the complex plane.
    
    Args:
        complex_numbers: List of complex numbers to plot
        labels: Optional labels for the complex numbers
        title: Plot title
        show_unit_circle: Whether to show the unit circle
        show_real_axis: Whether to highlight the real axis
        show_imag_axis: Whether to highlight the imaginary axis
        show_abs_value: Whether to show the absolute value (modulus)
        show_phase: Whether to show the phase (argument)
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for the figure
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with visualization information
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Extract real and imaginary parts
        real_parts = [complex(z).real for z in complex_numbers]
        imag_parts = [complex(z).imag for z in complex_numbers]
        
        # Set axis limits
        max_extent = max([abs(val) for val in real_parts + imag_parts]) * 1.2
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        
        # Plot unit circle if requested
        if show_unit_circle:
            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
            ax.add_artist(circle)
        
        # Plot axes
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight real and imaginary axes if requested
        if show_real_axis:
            ax.axhline(y=0, color='blue', linestyle='-', alpha=0.3, linewidth=2)
            ax.text(max_extent*0.9, 0.1, 'Real', ha='right', color='blue')
        
        if show_imag_axis:
            ax.axvline(x=0, color='red', linestyle='-', alpha=0.3, linewidth=2)
            ax.text(0.1, max_extent*0.9, 'Imaginary', va='top', color='red')
        
        # Plot complex numbers
        ax.scatter(real_parts, imag_parts, c='blue', s=100, zorder=5)
        
        # Add labels if provided
        if labels:
            for i, (x, y) in enumerate(zip(real_parts, imag_parts)):
                label = labels[i] if i < len(labels) else f"z{i+1}"
                offset_x = 0.1 if x >= 0 else -0.1
                offset_y = 0.1 if y >= 0 else -0.1
                ha = 'left' if x >= 0 else 'right'
                va = 'bottom' if y >= 0 else 'top'
                ax.text(x + offset_x, y + offset_y, label, ha=ha, va=va)
        
        # Show absolute value (modulus) if requested
        if show_abs_value:
            for i, z in enumerate(complex_numbers):
                r = abs(z)
                x, y = real_parts[i], imag_parts[i]
                
                # Draw line from origin to point
                ax.plot([0, x], [0, y], 'gray', linestyle='-', alpha=0.5)
                
                # Label with modulus value
                mid_x, mid_y = x/2, y/2
                ax.text(mid_x, mid_y, f"|z|={r:.2f}", 
                      ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Show phase (argument) if requested
        if show_phase:
            for i, z in enumerate(complex_numbers):
                x, y = real_parts[i], imag_parts[i]
                r = abs(z)
                
                if r > 0:  # Skip if at origin
                    theta = np.angle(z)
                    
                    # Draw arc representing the angle
                    arc_radius = min(0.5, r/2)
                    arc_angles = np.linspace(0, theta, 100)
                    arc_x = arc_radius * np.cos(arc_angles)
                    arc_y = arc_radius * np.sin(arc_angles)
                    ax.plot(arc_x, arc_y, 'green', alpha=0.7)
                    
                    # Label with angle value
                    angle_x = arc_radius * np.cos(theta/2)
                    angle_y = arc_radius * np.sin(theta/2)
                    angle_text = f"θ={theta*180/np.pi:.1f}°"
                    ax.text(angle_x*1.1, angle_y*1.1, angle_text, color='green',
                          ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Set equal aspect to ensure circles look circular
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add labels and title
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(title)
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "complex_numbers",
                "file_path": save_path,
                "data": {
                    "complex_numbers": [complex(z) for z in complex_numbers],
                    "real_parts": real_parts,
                    "imag_parts": imag_parts,
                    "moduli": [abs(z) for z in complex_numbers],
                    "phases": [np.angle(z) * 180 / np.pi for z in complex_numbers]
                }
            }
        else:
            # Convert to base64 for embedding in web applications
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            image_base64 = base64.b64encode(image_png).decode('utf-8')
            
            return {
                "success": True,
                "plot_type": "complex_numbers",
                "base64_image": image_base64,
                "data": {
                    "complex_numbers": [complex(z) for z in complex_numbers],
                    "real_parts": real_parts,
                    "imag_parts": imag_parts,
                    "moduli": [abs(z) for z in complex_numbers],
                    "phases": [np.angle(z) * 180 / np.pi for z in complex_numbers]
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to plot complex numbers: {str(e)}"
        }

def plot_vector_addition_2d(
    vectors: List[Tuple[float, float]],
    labels: Optional[List[str]] = None,
    title: str = "Vector Addition",
    origin: Tuple[float, float] = (0, 0),
    show_resultant: bool = True,
    colors: Optional[List[str]] = None,
    sequential: bool = False,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Visualize 2D vector addition.
    
    Args:
        vectors: List of 2D vectors as (x, y) tuples
        labels: Optional labels for the vectors
        title: Plot title
        origin: Starting point for vectors
        show_resultant: Whether to show the resultant vector
        colors: List of colors for the vectors
        sequential: If True, vectors are drawn head-to-tail; if False, all from origin
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for the figure
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with visualization information
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Default colors if not provided
        if colors is None:
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
            # Repeat colors if needed
            colors = colors * (len(vectors) // len(colors) + 1)
        
        # Create default labels if not provided
        if labels is None:
            labels = [f"v{i+1}" for i in range(len(vectors))]
        elif len(labels) < len(vectors):
            # Extend labels if needed
            labels.extend([f"v{i+1}" for i in range(len(labels), len(vectors))])
        
        # Store ending points for each vector
        endpoints = []
        
        # Track current origin
        current_origin = origin
        
        # Calculate resultant vector
        resultant = (0, 0)
        for vx, vy in vectors:
            resultant = (resultant[0] + vx, resultant[1] + vy)
        
        # Plot vectors
        for i, vector in enumerate(vectors):
            vx, vy = vector
            
            # Determine starting point
            start_x, start_y = current_origin
            
            # Calculate end point
            end_x, end_y = start_x + vx, start_y + vy
            endpoints.append((end_x, end_y))
            
            # Plot vector
            ax.arrow(start_x, start_y, vx, vy, head_width=0.1, head_length=0.2, 
                    fc=colors[i], ec=colors[i], length_includes_head=True)
            
            # Add label at midpoint
            mid_x, mid_y = start_x + vx/2, start_y + vy/2
            ax.text(mid_x, mid_y, labels[i], ha='center', va='bottom', color=colors[i])
            
            # Update current origin for sequential plotting
            if sequential:
                current_origin = (end_x, end_y)
        
        # Plot resultant vector if requested
        if show_resultant and len(vectors) > 0:
            if sequential:
                # For sequential, resultant goes from original origin to final point
                rx, ry = resultant
                ax.arrow(origin[0], origin[1], rx, ry, head_width=0.1, head_length=0.2, 
                        fc='black', ec='black', length_includes_head=True, linestyle='--')
                
                # Add label
                mid_x, mid_y = origin[0] + rx/2, origin[1] + ry/2
                ax.text(mid_x, mid_y, "Resultant", ha='center', va='top', color='black')
            else:
                # For non-sequential, all vectors start at origin, so resultant must be calculated
                rx, ry = resultant
                ax.arrow(origin[0], origin[1], rx, ry, head_width=0.1, head_length=0.2, 
                        fc='black', ec='black', length_includes_head=True, linestyle='--')
                
                # Add label
                mid_x, mid_y = origin[0] + rx/2, origin[1] + ry/2
                ax.text(mid_x, mid_y, "Resultant", ha='center', va='top', color='black')
        
        # Set equal aspect to ensure angles are preserved
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Determine axis limits
        all_x = [origin[0]] + [p[0] for p in endpoints]
        all_y = [origin[1]] + [p[1] for p in endpoints]
        
        # If showing resultant, include its endpoint
        if show_resultant:
            all_x.append(origin[0] + resultant[0])
            all_y.append(origin[1] + resultant[1])
        
        # Set axis limits with padding
        padding = 1.0
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Ensure equal padding on both sides
        x_range = max_x - min_x
        y_range = max_y - min_y
        max_range = max(x_range, y_range)
        
        # Set limits with padding
        ax.set_xlim(min_x - padding, min_x + max_range + padding)
        ax.set_ylim(min_y - padding, min_y + max_range + padding)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "vector_addition_2d",
                "file_path": save_path,
                "data": {
                    "vectors": vectors,
                    "resultant": resultant,
                    "sequential": sequential,
                    "endpoints": endpoints
                }
            }
        else:
            # Convert to base64 for embedding in web applications
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            image_base64 = base64.b64encode(image_png).decode('utf-8')
            
            return {
                "success": True,
                "plot_type": "vector_addition_2d",
                "base64_image": image_base64,
                "data": {
                    "vectors": vectors,
                    "resultant": resultant,
                    "sequential": sequential,
                    "endpoints": endpoints
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to plot vector addition: {str(e)}"
        }

def plot_linear_transformation_2d(
    transformation_matrix: List[List[float]],
    grid_lines: int = 10,
    grid_range: float = 4.0,
    title: str = "Linear Transformation",
    show_basis_vectors: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Visualize a 2D linear transformation with a grid.
    
    Args:
        transformation_matrix: 2x2 transformation matrix
        grid_lines: Number of grid lines in each direction
        grid_range: Range of the grid in each direction
        title: Plot title
        show_basis_vectors: Whether to highlight basis vectors
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for the figure
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with visualization information
    """
    try:
        # Validate transformation matrix
        if len(transformation_matrix) != 2 or len(transformation_matrix[0]) != 2:
            return {
                "success": False,
                "error": "Transformation matrix must be 2x2"
            }
        
        # Convert to numpy array
        A = np.array(transformation_matrix)
        
        # Create figure with two subplots (before and after)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Set grid range
        lim = grid_range
        
        # Generate grid points
        grid_step = 2 * lim / grid_lines
        x = np.arange(-lim, lim + grid_step, grid_step)
        y = np.arange(-lim, lim + grid_step, grid_step)
        X, Y = np.meshgrid(x, y)
        
        # Create grid line points
        x_lines = []
        y_lines = []
        
        for i in range(len(x)):
            x_lines.append(np.column_stack((np.full(len(y), x[i]), y)))
            y_lines.append(np.column_stack((x, np.full(len(x), y[i]))))
        
        # Plot original grid on the left
        for line in x_lines + y_lines:
            ax1.plot(line[:, 0], line[:, 1], 'b-', alpha=0.3)
        
        # Plot transformed grid on the right
        for line in x_lines + y_lines:
            # Apply transformation
            transformed = np.dot(line, A.T)
            ax2.plot(transformed[:, 0], transformed[:, 1], 'r-', alpha=0.3)
        
        # Plot basis vectors if requested
        if show_basis_vectors:
            # Original basis vectors
            ax1.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
            ax1.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.2, fc='green', ec='green')
            
            # Label original basis vectors
            ax1.text(1, -0.2, "e₁", color='blue')
            ax1.text(-0.2, 1, "e₂", color='green')
            
            # Transformed basis vectors
            transformed_e1 = A @ np.array([1, 0])
            transformed_e2 = A @ np.array([0, 1])
            
            ax2.arrow(0, 0, transformed_e1[0], transformed_e1[1], 
                    head_width=0.1, head_length=0.2, fc='blue', ec='blue')
            ax2.arrow(0, 0, transformed_e2[0], transformed_e2[1], 
                    head_width=0.1, head_length=0.2, fc='green', ec='green')
            
            # Label transformed basis vectors
            ax2.text(transformed_e1[0], transformed_e1[1] - 0.2, "T(e₁)", color='blue')
            ax2.text(transformed_e2[0] - 0.2, transformed_e2[1], "T(e₂)", color='green')
        
        # Set equal aspect ratios
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Calculate appropriate limits for transformed grid
        # Apply transformation to corner points
        corners = np.array([
            [-lim, -lim],
            [-lim, lim],
            [lim, -lim],
            [lim, lim]
        ])
        
        transformed_corners = np.dot(corners, A.T)
        
        # Set limits with padding
        padding = max(1.0, lim/5)
        
        ax1.set_xlim(-lim - padding, lim + padding)
        ax1.set_ylim(-lim - padding, lim + padding)
        
        min_x, max_x = transformed_corners[:, 0].min(), transformed_corners[:, 0].max()
        min_y, max_y = transformed_corners[:, 1].min(), transformed_corners[:, 1].max()
        
        # Calculate range and max range for equal aspect ratio
        x_range = max_x - min_x
        y_range = max_y - min_y
        max_range = max(x_range, y_range)
        
        # Center of the transformed grid
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Set limits with padding
        ax2.set_xlim(center_x - max_range/2 - padding, center_x + max_range/2 + padding)
        ax2.set_ylim(center_y - max_range/2 - padding, center_y + max_range/2 + padding)
        
        # Add titles and axis labels
        ax1.set_title("Original Space")
        ax2.set_title("Transformed Space")
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        fig.suptitle(title)
        
        # Create matrix text representation
        matrix_text = f"A = [[ {A[0, 0]:.2f}, {A[0, 1]:.2f} ],\n     [ {A[1, 0]:.2f}, {A[1, 1]:.2f} ]]"
        fig.text(0.5, 0.01, matrix_text, ha='center')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for matrix text
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "linear_transformation_2d",
                "file_path": save_path,
                "data": {
                    "transformation_matrix": transformation_matrix,
                    "determinant": float(np.linalg.det(A)),
                    "eigenvalues": np.linalg.eigvals(A).tolist()
                }
            }
        else:
            # Convert to base64 for embedding in web applications
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            image_base64 = base64.b64encode(image_png).decode('utf-8')
            
            return {
                "success": True,
                "plot_type": "linear_transformation_2d",
                "base64_image": image_base64,
                "data": {
                    "transformation_matrix": transformation_matrix,
                    "determinant": float(np.linalg.det(A)),
                    "eigenvalues": np.linalg.eigvals(A).tolist()
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to plot linear transformation: {str(e)}"
        }

def plot_probability_distribution(
    distribution_type: str,
    parameters: Dict[str, float],
    x_range: Optional[Tuple[float, float]] = None,
    num_points: int = 1000,
    title: Optional[str] = None,
    show_mean: bool = True,
    show_std_dev: bool = True,
    show_percentiles: bool = False,
    percentiles: List[float] = [5, 25, 50, 75, 95],
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot a probability distribution.
    
    Args:
        distribution_type: Type of distribution ('normal', 'uniform', 'exponential', etc.)
        parameters: Distribution parameters (e.g., {'mean': 0, 'std': 1} for normal)
        x_range: Range for the x-axis (if None, will be determined automatically)
        num_points: Number of points to sample
        title: Plot title (if None, will be generated based on distribution)
        show_mean: Whether to show the mean line
        show_std_dev: Whether to show standard deviation range
        show_percentiles: Whether to show percentile lines
        percentiles: Percentiles to show if show_percentiles is True
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for the figure
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with visualization information
    """
    try:
        # Import scipy stats
        from scipy import stats
        
        # Initialize distribution based on type
        dist = None
        stats_func = None
        
        # Handle different distribution types
        if distribution_type.lower() == 'normal' or distribution_type.lower() == 'gaussian':
            mean = parameters.get('mean', 0)
            std = parameters.get('std', 1)
            dist = stats.norm(loc=mean, scale=std)
            stats_func = lambda: {"mean": mean, "median": mean, "std": std, "variance": std**2, 
                                 "skewness": 0, "kurtosis": 0}
            
            # Set x_range if not provided
            if x_range is None:
                x_range = (mean - 4*std, mean + 4*std)
                
        elif distribution_type.lower() == 'uniform':
            a = parameters.get('a', 0)
            b = parameters.get('b', 1)
            dist = stats.uniform(loc=a, scale=b-a)
            mean = (a + b) / 2
            std = (b - a) / (12)**0.5
            stats_func = lambda: {"mean": mean, "median": mean, "std": std, 
                                 "variance": std**2, "a": a, "b": b}
            
            # Set x_range if not provided
            if x_range is None:
                padding = (b - a) * 0.1
                x_range = (a - padding, b + padding)
                
        elif distribution_type.lower() == 'exponential':
            lam = parameters.get('lambda', 1)
            dist = stats.expon(scale=1/lam)
            mean = 1/lam
            std = 1/lam
            stats_func = lambda: {"mean": mean, "median": np.log(2)/lam, "std": std, 
                                 "variance": std**2, "lambda": lam}
            
            # Set x_range if not provided
            if x_range is None:
                x_range = (0, 5/lam)
                
        elif distribution_type.lower() == 'poisson':
            mu = parameters.get('mu', 1)
            dist = stats.poisson(mu=mu)
            mean = mu
            std = mu**0.5
            stats_func = lambda: {"mean": mean, "median": int(mu + 1/3 - 0.02/mu), 
                                 "std": std, "variance": mu, "mu": mu}
            
            # Set x_range if not provided
            if x_range is None:
                x_range = (max(0, int(mean - 3*std)), int(mean + 3*std) + 1)
                
        elif distribution_type.lower() == 'binomial':
            n = parameters.get('n', 10)
            p = parameters.get('p', 0.5)
            dist = stats.binom(n=n, p=p)
            mean = n * p
            std = (n * p * (1-p))**0.5
            stats_func = lambda: {"mean": mean, "median": int(n*p), 
                                 "std": std, "variance": n*p*(1-p), "n": n, "p": p}
            
            # Set x_range if not provided
            if x_range is None:
                x_range = (0, n)
                
        else:
            return {
                "success": False,
                "error": f"Unsupported distribution type: {distribution_type}"
            }
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Generate x values based on range
        if distribution_type.lower() in ['poisson', 'binomial']:
            # Discrete distributions
            x = np.arange(x_range[0], x_range[1] + 1)
            
            # Calculate PMF
            y = dist.pmf(x)
            
            # Plot as bars
            ax.bar(x, y, alpha=0.7, width=0.4)
        else:
            # Continuous distributions
            x = np.linspace(x_range[0], x_range[1], num_points)
            
            # Calculate PDF
            y = dist.pdf(x)
            
            # Plot as line
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Fill under the curve
            ax.fill_between(x, y, alpha=0.2)
        
        # Get statistics
        stats_dict = stats_func()
        mean_val = stats_dict["mean"]
        std_val = stats_dict["std"]
        
        # Show mean if requested
        if show_mean:
            if distribution_type.lower() in ['poisson', 'binomial']:
                # For discrete distributions, use stem plot for mean
                ax.axvline(x=mean_val, color='red', linestyle='--', alpha=0.8)
            else:
                # For continuous distributions, use vertical line
                ax.axvline(x=mean_val, color='red', linestyle='--', alpha=0.8)
            
            # Add text for mean
            y_pos = ax.get_ylim()[1] * 0.9
            ax.text(mean_val, y_pos, f"Mean = {mean_val:.2f}", 
                   ha='center', va='top', color='red',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Show standard deviation range if requested
        if show_std_dev and distribution_type.lower() not in ['poisson', 'binomial']:
            # Only for continuous distributions
            lower = mean_val - std_val
            upper = mean_val + std_val
            
            # Highlight range
            ax.axvspan(lower, upper, alpha=0.2, color='green')
            
            # Add labels
            ax.axvline(x=lower, color='green', linestyle=':', alpha=0.8)
            ax.axvline(x=upper, color='green', linestyle=':', alpha=0.8)
            
            # Add text for std dev
            y_pos = ax.get_ylim()[1] * 0.8
            ax.text(mean_val, y_pos, f"±1σ = {std_val:.2f}", 
                   ha='center', va='top', color='green',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Show percentiles if requested
        if show_percentiles:
            percentile_colors = ['purple', 'orange', 'cyan', 'magenta', 'brown']
            
            for i, p in enumerate(percentiles):
                # Calculate percentile value
                p_val = dist.ppf(p/100)
                color = percentile_colors[i % len(percentile_colors)]
                
                # Draw line
                ax.axvline(x=p_val, color=color, linestyle=':', alpha=0.7)
                
                # Add label
                y_pos = ax.get_ylim()[1] * (0.7 - 0.05*i)
                ax.text(p_val, y_pos, f"{p}%", 
                       ha='center', va='top', color=color,
                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set labels
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density' if distribution_type.lower() not in ['poisson', 'binomial'] else 'Probability Mass')
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            if distribution_type.lower() == 'normal':
                title = f"Normal Distribution (μ={mean_val:.2f}, σ={std_val:.2f})"
            elif distribution_type.lower() == 'uniform':
                title = f"Uniform Distribution [a={parameters.get('a', 0)}, b={parameters.get('b', 1)}]"
            elif distribution_type.lower() == 'exponential':
                title = f"Exponential Distribution (λ={parameters.get('lambda', 1):.2f})"
            elif distribution_type.lower() == 'poisson':
                title = f"Poisson Distribution (μ={parameters.get('mu', 1):.2f})"
            elif distribution_type.lower() == 'binomial':
                title = f"Binomial Distribution (n={parameters.get('n', 10)}, p={parameters.get('p', 0.5):.2f})"
            else:
                title = f"{distribution_type.capitalize()} Distribution"
                
            ax.set_title(title)
        
        # Add summary statistics text box
        stats_text = "\n".join([f"{k}: {v:.4f}" for k, v in stats_dict.items()])
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "probability_distribution",
                "file_path": save_path,
                "data": {
                    "distribution_type": distribution_type,
                    "parameters": parameters,
                    "statistics": stats_dict
                }
            }
        else:
            # Convert to base64 for embedding in web applications
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            image_base64 = base64.b64encode(image_png).decode('utf-8')
            
            return {
                "success": True,
                "plot_type": "probability_distribution",
                "base64_image": image_base64,
                "data": {
                    "distribution_type": distribution_type,
                    "parameters": parameters,
                    "statistics": stats_dict
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to plot probability distribution: {str(e)}"
        }

def generate_unique_filename(base_dir: str, prefix: str = "plot", extension: str = "png") -> str:
    """Generate a unique filename for saving plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(base_dir, f"{prefix}_{timestamp}_{unique_id}.{extension}")

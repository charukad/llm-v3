import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
import io
import base64
import os
from datetime import datetime
import uuid

def plot_histogram(
    data: Union[List[float], np.ndarray],
    bins: Union[int, List[float], str] = 'auto',
    title: Optional[str] = None,
    x_label: str = "Value",
    y_label: str = "Frequency",
    figsize: Tuple[int, int] = (8, 6),
    color: str = '#1f77b4',
    edgecolor: str = 'black',
    density: bool = False,
    show_kde: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a histogram plot of numerical data.
    
    Args:
        data: List or array of numerical values
        bins: Number of bins, bin edges, or method to calculate bins
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        figsize: Figure size in inches (width, height)
        color: Fill color for bars
        edgecolor: Edge color for bars
        density: If True, the result is normalized to form a probability density
        show_kde: If True, shows a Kernel Density Estimate curve
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with plot information, including base64 encoded image or path
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Convert to numpy array if needed
        data_array = np.array(data)
        
        # Check for non-finite values
        finite_mask = np.isfinite(data_array)
        if not np.any(finite_mask):
            return {
                "success": False,
                "error": "No finite values in the data"
            }
        
        # Filter out non-finite values
        filtered_data = data_array[finite_mask]
        
        # Calculate basic statistics
        stats = {
            "mean": float(np.mean(filtered_data)),
            "median": float(np.median(filtered_data)),
            "std": float(np.std(filtered_data)),
            "min": float(np.min(filtered_data)),
            "max": float(np.max(filtered_data)),
            "count": int(len(filtered_data)),
            "non_finite": int(len(data_array) - len(filtered_data))
        }
        
        # Plot histogram
        n, bins_out, patches = ax.hist(
            filtered_data, 
            bins=bins, 
            color=color, 
            edgecolor=edgecolor, 
            alpha=0.7,
            density=density
        )
        
        # Add KDE if requested
        if show_kde and len(filtered_data) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(filtered_data)
            x_grid = np.linspace(min(filtered_data), max(filtered_data), 1000)
            ax.plot(x_grid, kde(x_grid), 'r-', linewidth=2)
        
        # Add mean line
        ax.axvline(stats["mean"], color='red', linestyle='dashed', linewidth=1)
        
        # Add text with statistics
        stats_text = (f"Mean: {stats['mean']:.2f}\n"
                     f"Median: {stats['median']:.2f}\n"
                     f"Std Dev: {stats['std']:.2f}\n"
                     f"Count: {stats['count']}")
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid, labels and title
        ax.grid(alpha=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Histogram")
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "histogram",
                "file_path": save_path,
                "data": {
                    "statistics": stats,
                    "bin_edges": bins_out.tolist(),
                    "bin_counts": n.tolist()
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
                "plot_type": "histogram",
                "base64_image": image_base64,
                "data": {
                    "statistics": stats,
                    "bin_edges": bins_out.tolist(),
                    "bin_counts": n.tolist()
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create histogram: {str(e)}"
        }

def plot_scatter(
    x_data: Union[List[float], np.ndarray],
    y_data: Union[List[float], np.ndarray],
    title: Optional[str] = None,
    x_label: str = "X",
    y_label: str = "Y",
    figsize: Tuple[int, int] = (8, 6),
    color: str = '#1f77b4',
    alpha: float = 0.7,
    show_regression: bool = False,
    marker: str = 'o',
    size: float = 30,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a scatter plot of two variables.
    
    Args:
        x_data: List or array of x values
        y_data: List or array of y values
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        figsize: Figure size in inches (width, height)
        color: Point color
        alpha: Transparency of points
        show_regression: If True, shows a linear regression line
        marker: Marker style
        size: Marker size
        save_path: Path to save the figure (if None, will be returned as base64)
        
    Returns:
        Dictionary with plot information, including base64 encoded image or path
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Convert to numpy arrays if needed
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # Check for matching lengths
        if len(x_array) != len(y_array):
            return {
                "success": False,
                "error": "X and Y data must have the same length"
            }
        
        # Check for non-finite values
        finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
        if not np.any(finite_mask):
            return {
                "success": False,
                "error": "No finite values in the data"
            }
        
        # Filter out non-finite values
        x_filtered = x_array[finite_mask]
        y_filtered = y_array[finite_mask]
        
        # Calculate basic statistics
        stats = {
            "x_mean": float(np.mean(x_filtered)),
            "y_mean": float(np.mean(y_filtered)),
            "x_std": float(np.std(x_filtered)),
            "y_std": float(np.std(y_filtered)),
            "count": int(len(x_filtered)),
            "non_finite": int(len(x_array) - len(x_filtered))
        }
        
        # Calculate correlation if possible
        if len(x_filtered) > 1:
            stats["correlation"] = float(np.corrcoef(x_filtered, y_filtered)[0, 1])
        
        # Plot scatter
        ax.scatter(
            x_filtered, 
            y_filtered, 
            color=color, 
            alpha=alpha,
            marker=marker,
            s=size
        )
        
        # Add regression line if requested
        if show_regression and len(x_filtered) > 1:
            from scipy import stats as spstats
            slope, intercept, r_value, p_value, std_err = spstats.linregress(x_filtered, y_filtered)
            
            # Add line
            x_line = np.linspace(min(x_filtered), max(x_filtered), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r-', linewidth=2)
            
            # Add regression information
            stats["regression"] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_err": float(std_err)
            }
            
            # Add text with regression info
            reg_text = (f"y = {slope:.4f}x + {intercept:.4f}\n"
                       f"R² = {r_value**2:.4f}\n"
                       f"p = {p_value:.4f}")
            
            ax.text(0.95, 0.05, reg_text,
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid, labels and title
        ax.grid(alpha=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Scatter Plot")
            
        # Add correlation info
        if "correlation" in stats:
            corr_text = f"Correlation: {stats['correlation']:.4f}"
            ax.text(0.95, 0.95, corr_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save or encode the figure
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            
            return {
                "success": True,
                "plot_type": "scatter",
                "file_path": save_path,
                "data": {
                    "statistics": stats
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
                "plot_type": "scatter",
                "base64_image": image_base64,
                "data": {
                    "statistics": stats
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create scatter plot: {str(e)}"
        }

def generate_unique_filename(base_dir: str, prefix: str = "plot", extension: str = "png") -> str:
    """Generate a unique filename for saving plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(base_dir, f"{prefix}_{timestamp}_{unique_id}.{extension}")

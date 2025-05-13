"""
Coordinate system detector for detecting and analyzing coordinate systems.

This module identifies coordinate axes, grids, and plots in images.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def detect_coordinate_system(image_path: str) -> Dict[str, Any]:
    """
    Detect coordinate systems in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing detected coordinate system
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "success": False,
                "error": f"Failed to read image: {image_path}",
                "has_coordinates": False
            }
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50, 
            minLineLength=50, maxLineGap=5
        )
        
        if lines is None:
            return {
                "success": True,
                "has_coordinates": False
            }
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle and length
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Classify lines based on angle
            if abs(angle) < 20 or abs(angle) > 160:
                horizontal_lines.append((x1, y1, x2, y2, length))
            elif abs(angle - 90) < 20 or abs(angle + 90) < 20:
                vertical_lines.append((x1, y1, x2, y2, length))
        
        # Sort lines by length (descending)
        horizontal_lines.sort(key=lambda x: x[4], reverse=True)
        vertical_lines.sort(key=lambda x: x[4], reverse=True)
        
        # Check if we have enough lines for a coordinate system
        has_coordinates = len(horizontal_lines) >= 1 and len(vertical_lines) >= 1
        
        if not has_coordinates:
            return {
                "success": True,
                "has_coordinates": False
            }
        
        # Find potential axes (longest lines)
        x_axis = horizontal_lines[0][:4] if horizontal_lines else None
        y_axis = vertical_lines[0][:4] if vertical_lines else None
        
        # Find potential origin (intersection of axes)
        origin = None
        if x_axis and y_axis:
            # Simple approach: find the closest points between the two lines
            x1, y1, x2, y2 = x_axis
            x3, y3, x4, y4 = y_axis
            
            # For simplicity, check distances between endpoints
            distances = [
                (np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2), (x1, y1)),
                (np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2), (x1, y1)),
                (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2), (x2, y2)),
                (np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2), (x2, y2))
            ]
            
            # Get the point with minimum distance
            min_distance, closest_point = min(distances)
            
            # If distance is small enough, consider it an intersection
            if min_distance < 30:
                origin = closest_point
        
        # Detect grid lines (shorter lines parallel to axes)
        grid_horizontal = []
        grid_vertical = []
        
        if len(horizontal_lines) > 1:
            for line in horizontal_lines[1:]:
                grid_horizontal.append(line[:4])
        
        if len(vertical_lines) > 1:
            for line in vertical_lines[1:]:
                grid_vertical.append(line[:4])
        
        # Determine plot area (bounding box of all lines)
        all_points = []
        for line in horizontal_lines + vertical_lines:
            all_points.append((line[0], line[1]))
            all_points.append((line[2], line[3]))
        
        if all_points:
            x_values = [p[0] for p in all_points]
            y_values = [p[1] for p in all_points]
            
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            
            plot_area = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        else:
            plot_area = None
        
        return {
            "success": True,
            "has_coordinates": True,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "origin": origin,
            "grid_horizontal": grid_horizontal,
            "grid_vertical": grid_vertical,
            "plot_area": plot_area
        }
        
    except Exception as e:
        logger.error(f"Error detecting coordinate system: {str(e)}")
        return {
            "success": False,
            "error": f"Error detecting coordinate system: {str(e)}",
            "has_coordinates": False
        }

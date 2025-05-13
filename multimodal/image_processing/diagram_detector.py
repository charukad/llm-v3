"""
Diagram detector for identifying and analyzing mathematical diagrams.

This module detects diagrams such as geometric shapes, plots, and
mathematical figures in images.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def detect_diagrams(image_path: str) -> Dict[str, Any]:
    """
    Detect diagrams in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing detected diagrams
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "success": False,
                "error": f"Failed to read image: {image_path}",
                "has_diagrams": False
            }
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50, 
            minLineLength=50, maxLineGap=10
        )
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        # Detect contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Initialize diagrams list
        diagrams = []
        
        # Check for geometric diagrams (shapes with clear contours)
        if contours:
            for contour in contours:
                # Calculate area and perimeter
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Skip small contours (likely noise)
                if area < 100:
                    continue
                
                # Approximate contour shape
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                vertices = len(approx)
                
                # Determine shape based on number of vertices
                shape_type = "unknown"
                if vertices == 3:
                    shape_type = "triangle"
                elif vertices == 4:
                    # Check if it's a square or rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.95 <= aspect_ratio <= 1.05:
                        shape_type = "square"
                    else:
                        shape_type = "rectangle"
                elif vertices == 5:
                    shape_type = "pentagon"
                elif vertices == 6:
                    shape_type = "hexagon"
                elif vertices > 6:
                    # Check if it's a circle by comparing area ratio
                    x, y, w, h = cv2.boundingRect(approx)
                    if w and h:
                        circle_area = np.pi * ((w + h) / 4) ** 2
                        if abs(area / circle_area - 1) < 0.2:
                            shape_type = "circle"
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add to diagrams list
                if shape_type != "unknown":
                    diagrams.append({
                        "type": "geometric",
                        "shape": shape_type,
                        "position": [int(x), int(y), int(w), int(h)],
                        "area": float(area),
                        "vertices": vertices
                    })
        
        # Check for line diagrams
        if lines is not None:
            line_groups = []
            
            # Group nearby lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                added_to_group = False
                
                for group in line_groups:
                    # Check if this line is close to any line in the group
                    for group_line in group:
                        gx1, gy1, gx2, gy2 = group_line
                        
                        # Simple check for line proximity
                        if (abs(x1 - gx1) < 20 and abs(y1 - gy1) < 20) or \
                           (abs(x2 - gx2) < 20 and abs(y2 - gy2) < 20):
                            group.append([x1, y1, x2, y2])
                            added_to_group = True
                            break
                            
                    if added_to_group:
                        break
                
                # If not added to any group, create a new group
                if not added_to_group:
                    line_groups.append([[x1, y1, x2, y2]])
            
            # Only consider groups with enough lines to be a diagram
            for group in line_groups:
                if len(group) >= 3:
                    # Find the bounding box of this line group
                    all_points = []
                    for line in group:
                        all_points.append([line[0], line[1]])
                        all_points.append([line[2], line[3]])
                    
                    all_points = np.array(all_points)
                    x_min, y_min = np.min(all_points, axis=0)
                    x_max, y_max = np.max(all_points, axis=0)
                    
                    # Add to diagrams list
                    diagrams.append({
                        "type": "line_diagram",
                        "position": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                        "line_count": len(group)
                    })
        
        # Determine if image has diagrams
        has_diagrams = len(diagrams) > 0
        
        return {
            "success": True,
            "has_diagrams": has_diagrams,
            "diagrams": diagrams
        }
        
    except Exception as e:
        logger.error(f"Error detecting diagrams: {str(e)}")
        return {
            "success": False,
            "error": f"Error detecting diagrams: {str(e)}",
            "has_diagrams": False
        }

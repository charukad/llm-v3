"""
Image Preprocessing for Mathematical Notation

This module provides image preprocessing functionality for handwritten mathematical notation,
including normalization, noise reduction, binarization, and other preprocessing steps
to prepare images for OCR.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Preprocesses images of handwritten mathematical notation for OCR.
    
    This class performs various preprocessing operations like normalization,
    noise reduction, binarization, and deskewing to optimize images for
    mathematical symbol recognition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config: Configuration parameters for preprocessing
        """
        self.config = config or {}
        
        # Default configuration values
        self.resize_size = self.config.get('resize_size', (1024, 1024))
        self.normalize = self.config.get('normalize', True)
        self.denoise = self.config.get('denoise', True)
        self.binarize = self.config.get('binarize', True)
        self.deskew = self.config.get('deskew', True)
        self.border = self.config.get('border', True)
        self.border_size = self.config.get('border_size', 10)
    
    def preprocess(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Apply preprocessing pipeline to an image.
        
        Args:
            image_data: Input image as numpy array
            
        Returns:
            Dictionary with processed image and metadata
        """
        try:
            # Create a copy of the input image
            processed_img = image_data.copy()
            
            # Initialize processing metadata
            metadata = {
                "original_shape": image_data.shape,
                "preprocessing_steps": []
            }
            
            # Check if the image is grayscale or color
            if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                # Convert to grayscale
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                metadata["preprocessing_steps"].append("convert_to_grayscale")
            
            # Apply preprocessing steps as configured
            if self.normalize:
                processed_img = self._normalize_image(processed_img)
                metadata["preprocessing_steps"].append("normalize")
            
            if self.resize_size:
                processed_img = self._resize_image(processed_img)
                metadata["preprocessing_steps"].append(f"resize_to_{self.resize_size}")
            
            if self.denoise:
                processed_img = self._reduce_noise(processed_img)
                metadata["preprocessing_steps"].append("denoise")
            
            if self.binarize:
                processed_img, threshold_value = self._binarize_image(processed_img)
                metadata["preprocessing_steps"].append(f"binarize_threshold_{threshold_value}")
            
            if self.deskew:
                angle = self._detect_skew(processed_img)
                if abs(angle) > 0.5:  # Only deskew if angle is significant
                    processed_img = self._deskew_image(processed_img, angle)
                    metadata["preprocessing_steps"].append(f"deskew_angle_{angle:.2f}")
                    metadata["deskew_angle"] = angle
            
            if self.border:
                processed_img = self._add_border(processed_img, self.border_size)
                metadata["preprocessing_steps"].append(f"add_border_{self.border_size}px")
            
            # Add final processed shape to metadata
            metadata["processed_shape"] = processed_img.shape
            
            # Return both the processed image and metadata
            return {
                "processed_image": processed_img,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            # Return original image if preprocessing fails
            return {
                "processed_image": image_data,
                "metadata": {
                    "original_shape": image_data.shape,
                    "preprocessing_steps": [],
                    "error": str(e)
                }
            }
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Normalized image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape
        target_w, target_h = self.resize_size
        
        # Calculate aspect ratio
        aspect = w / h
        
        # Determine new dimensions maintaining aspect ratio
        if w > h:
            new_w = min(w, target_w)
            new_h = int(new_w / aspect)
        else:
            new_h = min(h, target_h)
            new_w = int(new_h * aspect)
        
        # Ensure dimensions are not zero
        new_w = max(new_w, 1)
        new_h = max(new_h, 1)
        
        # Resize image
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to the image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        # Apply Non-local Means Denoising
        return cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
    
    def _binarize_image(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Convert image to binary using adaptive thresholding.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of binarized image and threshold value
        """
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        return binary, 127  # Return default threshold value with adaptive threshold
    
    def _detect_skew(self, image: np.ndarray) -> float:
        """
        Detect the skew angle of the text in the image.
        
        Args:
            image: Input binary image
            
        Returns:
            Detected skew angle in degrees
        """
        # Find all contours
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        
        for contour in contours:
            # Filter out very small contours
            if cv2.contourArea(contour) < 100:
                continue
                
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # Normalize angle to -45 to 45 degrees
            if angle < -45:
                angle += 90
            if angle > 45:
                angle -= 90
                
            angles.append(angle)
        
        # If no valid contours found, return 0
        if not angles:
            return 0
        
        # Use the median angle to avoid outliers
        return np.median(angles)
    
    def _deskew_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the image to correct skew.
        
        Args:
            image: Input image
            angle: Skew angle in degrees
            
        Returns:
            Deskewed image
        """
        # Get image dimensions
        h, w = image.shape
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate the image
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    def _add_border(self, image: np.ndarray, border_size: int) -> np.ndarray:
        """
        Add a white border around the image.
        
        Args:
            image: Input image
            border_size: Border width in pixels
            
        Returns:
            Image with border
        """
        return cv2.copyMakeBorder(
            image, 
            border_size, 
            border_size, 
            border_size, 
            border_size,
            cv2.BORDER_CONSTANT, 
            value=0 if np.mean(image) > 127 else 255
        )

def preprocess_image(image_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Preprocess an image from a file path.
    
    Args:
        image_path: Path to the image file
        config: Configuration parameters for preprocessing
        
    Returns:
        Dictionary with processed image and metadata
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to read image from {image_path}")
        
        # Create preprocessor and apply preprocessing
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.preprocess(image)
        
        # Add the original path to the metadata
        result["metadata"]["original_path"] = image_path
        
        return result
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise

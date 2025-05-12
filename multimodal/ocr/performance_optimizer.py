import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Any, Optional
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class OCRPerformanceOptimizer:
    """
    Optimizes performance of the mathematical OCR pipeline through
    various techniques like caching, parallel processing, and image preprocessing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance optimizer.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Caching parameters
        self.use_cache = self.config.get('use_cache', True)
        self.cache_size = self.config.get('cache_size', 100)
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Multithreading parameters
        self.use_threading = self.config.get('use_threading', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # Image optimization parameters
        self.max_image_size = self.config.get('max_image_size', 1600)
        self.target_dpi = self.config.get('target_dpi', 300)
        
        logger.info(f"Initialized OCRPerformanceOptimizer with cache_size={self.cache_size}, "
                   f"max_workers={self.max_workers}")
    
    def optimize_image(self, image) -> np.ndarray:
        """
        Optimize the image for faster processing.
        
        Args:
            image: Input image
            
        Returns:
            Optimized image
        """
        start_time = time.time()
        
        # Check if image is already optimized
        if self._is_optimized(image):
            logger.debug("Image already optimized, skipping")
            return image
        
        # Resize large images
        optimized = self._resize_if_needed(image)
        
        # Enhance contrast for better recognition
        optimized = self._enhance_contrast(optimized)
        
        # Apply specific optimizations for mathematical notation
        optimized = self._optimize_for_math(optimized)
        
        elapsed = time.time() - start_time
        logger.info(f"Image optimization completed in {elapsed:.3f}s")
        
        return optimized
    
    def parallel_process(self, 
                        image, 
                        regions: List[Dict[str, Any]], 
                        process_func) -> List[Dict[str, Any]]:
        """
        Process image regions in parallel.
        
        Args:
            image: Input image
            regions: List of regions to process
            process_func: Function to apply to each region
            
        Returns:
            List of processed results
        """
        if not regions:
            return []
        
        if not self.use_threading or len(regions) <= 1:
            # Process sequentially for small number of regions
            return [process_func(image, region) for region in regions]
        
        # Process regions in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_func, image, region): i 
                for i, region in enumerate(regions)
            }
            
            for future in as_completed(futures):
                results.append((futures[future], future.result()))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Get a result from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        if not self.use_cache:
            return None
        
        with self.cache_lock:
            result = self.cache.get(key)
            if result:
                logger.debug(f"Cache hit for key {key}")
                # Update access time
                result['last_accessed'] = time.time()
                return result['data']
            
            logger.debug(f"Cache miss for key {key}")
            return None
    
    def cache_result(self, key: str, data: Any) -> None:
        """
        Cache a result.
        
        Args:
            key: Cache key
            data: Result to cache
        """
        if not self.use_cache:
            return
        
        with self.cache_lock:
            # Check cache size
            if len(self.cache) >= self.cache_size:
                self._prune_cache()
            
            # Add to cache
            self.cache[key] = {
                'data': data,
                'created': time.time(),
                'last_accessed': time.time()
            }
            logger.debug(f"Cached result for key {key}")
    
    def _is_optimized(self, image) -> bool:
        """Check if an image is already optimized."""
        # Implementation-specific check
        # For example, check if image size is already within limits
        height, width = image.shape[:2]
        return max(height, width) <= self.max_image_size
    
    def _resize_if_needed(self, image) -> np.ndarray:
        """Resize image if needed."""
        height, width = image.shape[:2]
        
        if max(height, width) > self.max_image_size:
            # Calculate new dimensions
            if width > height:
                new_width = self.max_image_size
                new_height = int(height * (self.max_image_size / width))
            else:
                new_height = self.max_image_size
                new_width = int(width * (self.max_image_size / height))
            
            # Resize
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized
        
        return image
    
    def _enhance_contrast(self, image) -> np.ndarray:
        """Enhance image contrast for better OCR."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def _optimize_for_math(self, image) -> np.ndarray:
        """Apply optimizations specific for mathematical notation."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert back to black text on white background
        result = cv2.bitwise_not(cleaned)
        
        return result
    
    def _prune_cache(self) -> None:
        """Remove oldest or least recently used items from cache."""
        # Sort by last accessed time
        sorted_items = sorted(
            self.cache.items(), 
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest 10% of items
        items_to_remove = max(1, int(len(self.cache) * 0.1))
        for i in range(items_to_remove):
            if i < len(sorted_items):
                del self.cache[sorted_items[i][0]]
        
        logger.debug(f"Pruned {items_to_remove} items from cache")

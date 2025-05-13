"""
PDF processing utilities.

This module provides utilities for processing PDF files, with fallbacks
when PyMuPDF is not available.
"""
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Check if PyMuPDF is available
PYMUPDF_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF not available. PDF handling will be limited.")

def extract_pdf_text(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of text content by page
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available. Cannot extract text from PDF.")
        return ["[PDF TEXT EXTRACTION NOT AVAILABLE]"]
    
    try:
        pdf_document = fitz.open(pdf_path)
        results = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            results.append(text)
        
        return results
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return []

def extract_pdf_images(pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Extract images from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional directory to save extracted images
        
    Returns:
        List of paths to extracted images
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available. Cannot extract images from PDF.")
        return []
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        pdf_document = fitz.open(pdf_path)
        image_paths = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Get images
            image_list = page.get_images(full=True)
            
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save the image
                img_filename = f"page{page_num+1}_img{img_idx+1}.{base_image['ext']}"
                img_path = os.path.join(output_dir, img_filename)
                
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_paths.append(img_path)
        
        return image_paths
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
        return []

def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Get metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary of metadata
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available. Cannot extract metadata from PDF.")
        return {"error": "PDF metadata extraction not available"}
    
    try:
        pdf_document = fitz.open(pdf_path)
        metadata = pdf_document.metadata
        
        # Add page count
        metadata["page_count"] = len(pdf_document)
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from PDF: {e}")
        return {"error": str(e)} 
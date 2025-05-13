"""
Format handler for detecting and converting image formats.

This module handles various file formats including images, PDFs,
and other document formats.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import os
import tempfile
import mimetypes
import shutil

logger = logging.getLogger(__name__)

# Import optional dependencies if available
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. PDF handling will be limited.")

def detect_format(file_path: str) -> Dict[str, Any]:
    """
    Detect the format of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing format information
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        # Get file extension and mime type
        _, ext = os.path.splitext(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Initialize format info
        format_info = {
            "success": True,
            "file_path": file_path,
            "extension": ext.lower(),
            "mime_type": mime_type
        }
        
        # Categorize format
        if mime_type:
            if mime_type.startswith("image/"):
                format_info["format"] = "standard" if mime_type in ["image/png", "image/jpeg", "image/jpg"] else "image"
            elif mime_type == "application/pdf":
                format_info["format"] = "pdf"
                
                # Get additional PDF info if PyMuPDF is available
                if PYMUPDF_AVAILABLE:
                    try:
                        doc = fitz.open(file_path)
                        format_info["pages"] = len(doc)
                        format_info["metadata"] = doc.metadata
                    except Exception as e:
                        logger.warning(f"Error getting PDF info: {str(e)}")
            else:
                format_info["format"] = "unknown"
        else:
            format_info["format"] = "unknown"
        
        return format_info
        
    except Exception as e:
        logger.error(f"Error detecting format: {str(e)}")
        return {
            "success": False,
            "error": f"Error detecting format: {str(e)}"
        }

def convert_format(file_path: str, format_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert a file to a standard image format.
    
    Args:
        file_path: Path to the file
        format_info: Optional format information from detect_format
        
    Returns:
        Dictionary containing conversion results
    """
    try:
        # Detect format if not provided
        if not format_info:
            format_info = detect_format(file_path)
            
            if not format_info.get("success", False):
                return format_info
        
        format_type = format_info.get("format")
        
        # If already standard format, just return the path
        if format_type == "standard":
            return {
                "success": True,
                "converted_path": file_path,
                "original_path": file_path,
                "conversion_needed": False
            }
        
        # Handle PDF conversion
        if format_type == "pdf" and PYMUPDF_AVAILABLE:
            # Create temporary directory for extracted images
            temp_dir = tempfile.mkdtemp()
            converted_paths = []
            
            try:
                # Open the PDF
                doc = fitz.open(file_path)
                
                # Extract each page as an image
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # Render page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    output_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    pix.save(output_path)
                    
                    converted_paths.append(output_path)
                
                # If only one page, return that path
                if len(converted_paths) == 1:
                    return {
                        "success": True,
                        "converted_path": converted_paths[0],
                        "original_path": file_path,
                        "conversion_needed": True,
                        "format": "image/png"
                    }
                else:
                    # Return all paths for multi-page documents
                    return {
                        "success": True,
                        "converted_paths": converted_paths,
                        "converted_path": converted_paths[0],  # Default to first page
                        "original_path": file_path,
                        "conversion_needed": True,
                        "multi_page": True,
                        "format": "image/png",
                        "pages": len(converted_paths)
                    }
                    
            except Exception as e:
                # Clean up temp directory on error
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise e
        
        # Handle other image formats (using simple copy for now)
        elif format_type == "image":
            # Create temporary file with standard extension
            _, temp_path = tempfile.mkstemp(suffix=".png")
            
            # Copy file content
            shutil.copyfile(file_path, temp_path)
            
            return {
                "success": True,
                "converted_path": temp_path,
                "original_path": file_path,
                "conversion_needed": True,
                "format": "image/png"
            }
        
        # Format not supported for conversion
        return {
            "success": False,
            "error": f"Unsupported format for conversion: {format_type}",
            "original_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Error converting format: {str(e)}")
        return {
            "success": False,
            "error": f"Error converting format: {str(e)}",
            "original_path": file_path
        }

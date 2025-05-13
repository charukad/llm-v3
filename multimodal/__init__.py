"""
Multimodal Input Processing for Mathematical Content

This package provides the components for processing various forms of
mathematical input, including handwritten notation, diagrams, and
coordinate systems.
"""

from multimodal.image_processing import (
    ImagePreprocessor, 
    detect_diagrams, 
    detect_coordinate_system
)
from multimodal.image_processing.format_handler import (
    detect_format,
    convert_format
)
from multimodal.ocr import (
    detect_symbols,
    detect_advanced_symbols,
    MathContextAnalyzer,
    OCRPerformanceOptimizer
)
from multimodal.structure import (
    analyze_layout
)
from multimodal.latex_generator import (
    generate_latex
)
from multimodal.agent import (
    OCRAgent,
    AdvancedOCRAgent
)

__version__ = '0.2.0'  # Updated for Sprint 11

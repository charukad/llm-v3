"""
Multimodal Input Processing for Mathematical Content

This package provides the components for processing various forms of
mathematical input, including handwritten notation, diagrams, and
coordinate systems.
"""

from multimodal.image_processing import (
    ImagePreprocessor, 
    FormatHandler, 
    DiagramDetector, 
    CoordinateSystemDetector
)
from multimodal.ocr import (
    SymbolDetector,
    MathSymbolDetector,
    MathContextAnalyzer,
    OCRPerformanceOptimizer
)
from multimodal.structure import (
    MathLayoutAnalyzer
)
from multimodal.latex_generator import (
    LaTeXGenerator
)
from multimodal.agent import (
    HandwritingRecognitionAgent,
    AdvancedOCRAgent
)

__version__ = '0.2.0'  # Updated for Sprint 11

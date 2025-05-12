"""
Multimodal Configuration for the Mathematical Multimodal LLM System.

This module provides configuration settings for multimodal operations,
including handwriting recognition, image processing, and visualization.
"""
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field


class OCRSettings(BaseModel):
    """Settings for OCR processing."""
    min_confidence_threshold: float = 0.7
    enhance_image: bool = True
    detect_diagrams: bool = True
    detect_coordinates: bool = True
    max_image_size: int = 4000  # Maximum dimension in pixels
    supported_formats: List[str] = ["jpg", "jpeg", "png", "bmp", "tiff"]
    symbol_detection_model: str = "math_symbol_detector_v2"
    layout_analysis_model: str = "math_layout_analyzer_v1"
    ocr_batch_size: int = 16
    use_gpu: bool = True


class VisualizationSettings(BaseModel):
    """Settings for visualization generation."""
    default_width: int = 800
    default_height: int = 600
    default_dpi: int = 100
    default_format: str = "png"
    max_points: int = 10000  # Maximum points for plotting
    enable_interactive: bool = True
    color_scheme: str = "viridis"
    font_family: str = "DejaVu Sans"
    include_grid: bool = True
    supported_formats: List[str] = ["png", "svg", "pdf", "jpg"]
    use_tex: bool = True  # Use TeX for rendering labels


class MultimodalConfig(BaseModel):
    """Complete multimodal configuration settings."""
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    enable_handwritten_input: bool = True
    enable_diagram_recognition: bool = True
    enable_3d_visualization: bool = True
    enable_interactive_visualization: bool = True
    max_image_upload_size_mb: int = 10
    supported_input_formats: List[str] = ["jpg", "jpeg", "png", "pdf", "bmp", "tiff"]
    supported_output_formats: List[str] = ["png", "svg", "pdf", "jpg"]


# Create default configuration
DEFAULT_CONFIG = MultimodalConfig()

# Optimization presets
LOW_RESOURCE_CONFIG = MultimodalConfig(
    ocr=OCRSettings(
        enhance_image=False,
        detect_diagrams=False,
        detect_coordinates=False,
        max_image_size=2000,
        ocr_batch_size=4,
        use_gpu=False
    ),
    visualization=VisualizationSettings(
        default_width=600,
        default_height=450,
        default_dpi=72,
        max_points=5000,
        enable_interactive=False,
        use_tex=False
    ),
    enable_3d_visualization=False,
    enable_interactive_visualization=False,
    max_image_upload_size_mb=5
)

HIGH_PERFORMANCE_CONFIG = MultimodalConfig(
    ocr=OCRSettings(
        min_confidence_threshold=0.6,
        enhance_image=True,
        detect_diagrams=True,
        detect_coordinates=True,
        max_image_size=8000,
        ocr_batch_size=32,
        use_gpu=True
    ),
    visualization=VisualizationSettings(
        default_width=1200,
        default_height=900,
        default_dpi=150,
        max_points=50000,
        enable_interactive=True,
        use_tex=True
    ),
    enable_3d_visualization=True,
    enable_interactive_visualization=True,
    max_image_upload_size_mb=50
)


# Current active configuration (singleton)
_active_config = DEFAULT_CONFIG

def get_multimodal_config() -> MultimodalConfig:
    """Get the current active multimodal configuration."""
    return _active_config

def set_multimodal_config(config: MultimodalConfig):
    """Set the active multimodal configuration."""
    global _active_config
    _active_config = config

def use_optimization_preset(preset: str):
    """Use a predefined optimization preset."""
    global _active_config
    
    if preset.lower() == "low_resource":
        _active_config = LOW_RESOURCE_CONFIG
    elif preset.lower() == "high_performance":
        _active_config = HIGH_PERFORMANCE_CONFIG
    elif preset.lower() == "default":
        _active_config = DEFAULT_CONFIG
    else:
        raise ValueError(f"Unknown optimization preset: {preset}")

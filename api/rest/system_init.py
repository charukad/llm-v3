"""
System Initialization for the Mathematical Multimodal LLM System.

This module handles the initialization of all system services and agents.
"""
import os
import logging
from typing import Dict, Any
import torch

from orchestration.agents.registry import get_agent_registry
from multimodal.context.context_manager import ContextManager
from multimodal.unified_pipeline.input_processor import InputProcessor
from multimodal.unified_pipeline.content_router import ContentRouter
from multimodal.agent.ocr_agent import OCRAgent
from multimodal.agent.advanced_ocr_agent import AdvancedOCRAgent
from core.agent.llm_agent import CoreLLMAgent
from math_processing.agent.math_agent import MathComputationAgent
from multimodal.interaction.ambiguity_handler import AmbiguityHandler
from multimodal.interaction.feedback_processor import FeedbackProcessor

logger = logging.getLogger(__name__)

# Get agent registry singleton
registry = get_agent_registry()

def initialize_system():
    """Initialize all system components."""
    try:
        # Check for GPU acceleration support
        check_acceleration_support()
        
        # Initialize services
        initialize_services()
        
        # Initialize agents
        initialize_agents()
        
        # Initialize MongoDB (optional)
        initialize_mongodb()
        
        logger.info("System initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during system initialization: {e}")
        logger.error("Some system components may not be available")
        return False

def check_acceleration_support():
    """Check for GPU support and LMStudio connectivity."""
    # Check for CUDA support (informational only)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        logger.info(f"CUDA is available: {device_count} device(s) detected")
        logger.info(f"Primary GPU: {device_name}")
    else:
        logger.info("CUDA is not available. Running in CPU-only mode.")
    
    # Check LMStudio connectivity (primary inference method)
    lmstudio_url = os.environ.get('LMSTUDIO_URL', 'http://127.0.0.1:1234')
    lmstudio_enabled = os.environ.get('USE_LMSTUDIO', '1') == '1'
    
    if lmstudio_enabled:
        try:
            import requests
            response = requests.get(f"{lmstudio_url}/v1/models", timeout=2)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"LMStudio connection successful. Available models: {models}")
                os.environ["LMSTUDIO_CONNECTED"] = "1"
            else:
                logger.warning(f"LMStudio responded with unexpected status code: {response.status_code}")
                os.environ["LMSTUDIO_CONNECTED"] = "0"
        except Exception as e:
            logger.warning(f"Could not connect to LMStudio at {lmstudio_url}: {e}")
            os.environ["LMSTUDIO_CONNECTED"] = "0"
    else:
        logger.info("LMStudio integration is disabled by configuration.")

def initialize_services():
    """Initialize and register services needed by the system."""
    # Create service instances
    context_manager = ContextManager()
    input_processor = InputProcessor()
    content_router = ContentRouter()
    ocr_agent = OCRAgent()
    advanced_ocr_agent = AdvancedOCRAgent()
    ambiguity_handler = AmbiguityHandler()
    feedback_processor = FeedbackProcessor()
    
    # Register services with the registry
    registry.register_service(
        service_id="context_manager",
        service_info={
            "name": "Context Manager",
            "instance": context_manager
        }
    )
    
    registry.register_service(
        service_id="input_processor",
        service_info={
            "name": "Input Processor",
            "instance": input_processor
        }
    )
    
    registry.register_service(
        service_id="content_router",
        service_info={
            "name": "Content Router",
            "instance": content_router
        }
    )
    
    registry.register_service(
        service_id="ocr_agent",
        service_info={
            "name": "OCR Agent",
            "instance": ocr_agent
        }
    )
    
    registry.register_service(
        service_id="advanced_ocr_agent",
        service_info={
            "name": "Advanced OCR Agent",
            "instance": advanced_ocr_agent
        }
    )
    
    registry.register_service(
        service_id="ambiguity_handler",
        service_info={
            "name": "Ambiguity Handler",
            "instance": ambiguity_handler
        }
    )
    
    registry.register_service(
        service_id="feedback_processor",
        service_info={
            "name": "Feedback Processor",
            "instance": feedback_processor
        }
    )
    
    logger.info("Initialized system services")

def initialize_agents():
    """Initialize agent instances and update registry with them."""
    # Configure LLM agent with LMStudio settings
    llm_config = {
        "use_lmstudio": os.environ.get('USE_LMSTUDIO', '1') == '1',
        "lmstudio_url": os.environ.get('LMSTUDIO_URL', 'http://127.0.0.1:1234'),
        "lmstudio_model": os.environ.get('LMSTUDIO_MODEL', 'mistral-7b-instruct-v0.3'),
        "use_vllm": False  # Disable vLLM since we're using LMStudio
    }
    
    # Log the configuration
    logger.info(f"LLM Agent config: LMStudio enabled: {llm_config['use_lmstudio']}, URL: {llm_config['lmstudio_url']}, Model: {llm_config['lmstudio_model']}")
    
    # Create agent instances
    core_llm_agent = CoreLLMAgent(config=llm_config)
    math_agent = MathComputationAgent()
    
    # OCR agents are already initialized in services
    ocr_agent_instance = None
    advanced_ocr_instance = None
    if "ocr_agent" in registry.services:
        ocr_agent_instance = registry.services["ocr_agent"].get("instance")
    if "advanced_ocr_agent" in registry.services:
        advanced_ocr_instance = registry.services["advanced_ocr_agent"].get("instance")
    
    # Register agent instances
    if "core_llm_agent" in registry.agents:
        registry.agents["core_llm_agent"]["instance"] = core_llm_agent
        logger.info("Registered CoreLLMAgent instance")
    
    if "math_computation_agent" in registry.agents:
        registry.agents["math_computation_agent"]["instance"] = math_agent
        logger.info("Registered MathComputationAgent instance")
    
    if "ocr_agent" in registry.agents and ocr_agent_instance:
        registry.agents["ocr_agent"]["instance"] = ocr_agent_instance
        logger.info("Registered OCRAgent instance")
    
    # Handle visualization agent - optional component
    if "visualization_agent" in registry.agents:
        try:
            from visualization.agent import VisualizationAgent
            viz_agent = VisualizationAgent()
            registry.agents["visualization_agent"]["instance"] = viz_agent
            logger.info("Registered VisualizationAgent instance")
        except ImportError:
            # Visualization module has been removed/not available
            logger.info("Visualization module not available - skipping initialization")
    
    # Handle search agent - optional component
    if "search_agent" in registry.agents:
        try:
            from search.agent import SearchAgent
            search_agent = SearchAgent()
            registry.agents["search_agent"]["instance"] = search_agent
            logger.info("Registered SearchAgent instance")
        except ImportError:
            # Search module not available
            logger.info("Search module not available - skipping initialization")

def initialize_mongodb():
    """Initialize MongoDB connection if available."""
    try:
        # Only import MongoDB modules if needed
        from database.access.mongodb_wrapper import MongoDBWrapper
        
        # Get MongoDB URI from environment variable
        mongodb_uri = os.environ.get('MONGODB_URI')
        
        if not mongodb_uri:
            logger.info("MONGODB_URI environment variable not set. Using default connection.")
            mongodb_uri = "mongodb://localhost:27017/math_llm_system"
        
        # Initialize MongoDB connection
        try:
            mongodb_wrapper = MongoDBWrapper(mongodb_uri)
            logger.info(f"Connecting to MongoDB: {mongodb_uri.split('/')[-1]}")
            
            # Check if connection is successful
            if mongodb_wrapper.client:
                # Get database name from wrapper
                db_name = getattr(mongodb_wrapper, 'database_name', 'math_llm_system')
                logger.info(f"Connected to MongoDB: {db_name}")
                
                # Register in the registry
                registry.register_service(
                    service_id="mongodb",
                    service_info={
                        "name": "MongoDB Database",
                        "instance": mongodb_wrapper
                    }
                )
                logger.info("Registered MongoDB service")
                return True
            else:
                logger.info("Failed to connect to MongoDB")
                return False
        except Exception as e:
            logger.info(f"MongoDB connection attempt failed: {e}")
            logger.info("Continuing without MongoDB support")
            return False
    except ImportError:
        logger.info("MongoDB modules not available. Database functionality will be limited.")
    except Exception as e:
        logger.info(f"MongoDB initialization skipped: {e}")
    
    return False 
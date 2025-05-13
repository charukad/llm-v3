"""
API routes for the Mathematical Multimodal LLM System.
"""
import logging

logger = logging.getLogger(__name__)

# Define all routers list
routers = []

# Import basic routes
from api.rest.routes.math import router as math_router
from api.rest.routes.multimodal import router as multimodal_router
from api.rest.routes.visualization import router as visualization_router

# Add basic routers
routers.append(math_router)
routers.append(multimodal_router)
routers.append(visualization_router)

# Try to import workflow router
try:
    from api.rest.routes.workflow import router as workflow_router
    routers.append(workflow_router)
    logger.info("Workflow router loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import workflow router: {e}")
except Exception as e:
    logger.error(f"Error importing workflow router: {e}")

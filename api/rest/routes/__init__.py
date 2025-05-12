"""
Route definitions for the REST API.
"""

# Import routes to make them available
from api.rest.routes.math import router as math_router
from api.rest.routes.multimodal import router as multimodal_router
from api.rest.routes.visualization import router as visualization_router
from api.rest.routes.workflow import router as workflow_router

# List of all routers to be registered
routers = [
    math_router,
    multimodal_router,
    visualization_router,
    workflow_router
]

#!/usr/bin/env python3
"""
Simple FastAPI application to directly test the LLM functionality.
This bypasses the workflow system to verify the LLM is working correctly.
"""
import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.agent.llm_agent import CoreLLMAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LMStudio Direct Test API",
    description="Simple API to test LMStudio integration directly"
)

# Initialize CoreLLMAgent
llm_agent = CoreLLMAgent()

# Define request model
class MathQuery(BaseModel):
    query: str
    use_cot: bool = True

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "name": "LMStudio Direct Test API",
        "status": "running",
        "llm_config": {
            "use_lmstudio": True,
            "lmstudio_url": os.environ.get('LMSTUDIO_URL', 'http://127.0.0.1:1234'),
            "lmstudio_model": os.environ.get('LMSTUDIO_MODEL', 'mistral-7b-instruct-v0.3')
        }
    }

@app.post("/solve")
def solve_math_problem(query_request: MathQuery):
    """Solve a mathematical problem directly using the LLM."""
    logger.info(f"Received query: {query_request.query}")
    
    try:
        # Generate response directly from the LLM
        result = llm_agent.generate_response(
            prompt=query_request.query, 
            use_cot=query_request.use_cot
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "query": query_request.query,
                "response": result.get("response", ""),
                "processing_time_ms": result.get("processing_time_ms", 0)
            }
        else:
            logger.error(f"LLM generation error: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get("error", "LLM generation failed"))
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting LMStudio Direct Test API")
    uvicorn.run(app, host="0.0.0.0", port=8002) 
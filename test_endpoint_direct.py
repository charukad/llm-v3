#!/usr/bin/env python3
"""
Direct test for the AI Analysis endpoint.
"""
import asyncio
import json
import logging
import aiohttp
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("endpoint_test")

async def test_analysis_endpoint(query):
    """Test the analysis endpoint with a given query."""
    url = "http://localhost:8000/ai-analysis/analyze"
    
    async with aiohttp.ClientSession() as session:
        logger.info(f"Sending query to endpoint: {query}")
        
        try:
            async with session.post(
                url,
                json={"query": query},
                timeout=60  # 60 second timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Response status: {response.status}")
                    logger.info(f"Response: {json.dumps(result, indent=2)}")
                    return result
                else:
                    logger.error(f"Error: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return None
        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
            return None

async def main():
    """Main function."""
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "What is 2+235?"
    
    logger.info("Starting AI Analysis Endpoint Test")
    result = await test_analysis_endpoint(query)
    
    if result:
        logger.info("Test completed successfully")
        if result.get("analysis", {}).get("ai_source") == "real_core_llm":
            logger.info("✅ SUCCESS: Response came from real CoreLLM")
        else:
            logger.info("⚠️ WARNING: Response came from fallback mechanism")
    else:
        logger.error("Test failed")

if __name__ == "__main__":
    asyncio.run(main()) 
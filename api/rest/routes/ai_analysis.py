"""
API routes for AI-powered mathematical query analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
import logging
import time
import asyncio
import json
import re
import traceback

router = APIRouter(prefix="/ai-analysis", tags=["ai-analysis"])

# Initialize logger
logger = logging.getLogger(__name__)


class RealAIAnalysisAgent:
    """
    Real AI analysis agent that uses the CoreLLM to analyze queries.
    """
    
    def __init__(self):
        """Initialize the real AI analysis agent."""
        self.llm_agent = None
        
        # Analysis prompt template
        self.analysis_prompt_template = """
        You are a specialized AI for analyzing mathematical queries. Your task is to analyze the following mathematical query and provide a detailed breakdown of what's needed to solve it.
        
        For the query: "{query}"
        
        Analyze and determine:
        1. The types of mathematical operations required (e.g., equation_solving, differentiation, integration, plotting)
        2. The mathematical concepts involved (e.g., algebra, calculus, trigonometry, statistics)
        3. The specialized agents that would be needed to process this query:
           - core_llm_agent: For query classification, response generation, math explanation
           - math_computation_agent: For equation solving, calculus, linear algebra, statistics
           - visualization_agent: For plotting functions, statistical visualizations, 3D plots
           - search_agent: For retrieving mathematical references or additional information
        4. The complexity level (simple, moderate, complex)
        5. Any sub-problems that need to be solved separately
        
        Provide your analysis in a valid JSON format with the following structure:
        {{
            "operations": ["operation1", "operation2", ...],
            "concepts": ["concept1", "concept2", ...],
            "required_agents": ["agent1", "agent2", ...],
            "complexity": "simple|moderate|complex",
            "sub_problems": [
                {{"type": "sub_problem_type", "description": "sub_problem_description"}},
                ...
            ],
            "routing": {{
                "primary_agent": "the_main_agent_needed",
                "confidence": 0.XX,
                "alternative_agents": ["agent1", "agent2"]
            }}
        }}
        
        Make sure the JSON is valid and properly formatted. Only return the JSON object, nothing else.
        """
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a mathematical query using the CoreLLM.
        
        Args:
            query: The mathematical query to analyze
            
        Returns:
            Analysis results
        """
        try:
            if self.llm_agent is None:
                from api.rest.server import get_core_llm_agent
                self.llm_agent = get_core_llm_agent()
                logger.info(f"LLM agent initialized: {self.llm_agent is not None}")
            
            if not self.llm_agent:
                logger.error("CoreLLM agent not available for analysis")
                return self._get_fallback_analysis(query, "CoreLLM agent not available")
            
            # Format the prompt with the query
            prompt = self.analysis_prompt_template.format(query=query)
            
            # Get response from CoreLLM
            try:
                # First, try the known CoreLLMAgent method: generate_response
                if hasattr(self.llm_agent, 'generate_response'):
                    logger.info("Using generate_response method to call CoreLLM")
                    # Add timeout to prevent hanging
                    try:
                        # Create a task with timeout
                        response_task = asyncio.create_task(self._run_with_timeout(prompt))
                        response_dict = await asyncio.wait_for(response_task, timeout=90.0)  # 90 second timeout (increased from 30)
                    except asyncio.TimeoutError:
                        logger.error("LLM response timed out after 90 seconds")
                        return self._get_fallback_analysis(query, "LLM response timed out")
                        
                    if isinstance(response_dict, dict) and "response" in response_dict:
                        response_text = response_dict["response"]
                    else:
                        response_text = str(response_dict)
                # Try different ways to call the CoreLLM based on its API as fallbacks
                elif hasattr(self.llm_agent, 'generate'):
                    logger.info("Using generate method to call CoreLLM")
                    response = self.llm_agent.generate(prompt)
                    response_text = response
                elif hasattr(self.llm_agent, 'generate_text'):
                    logger.info("Using generate_text method to call CoreLLM")
                    response = self.llm_agent.generate_text(prompt, temperature=0.1, max_tokens=1024)
                    response_text = response
                elif hasattr(self.llm_agent, 'process_prompt'):
                    logger.info("Using process_prompt method to call CoreLLM")
                    response = self.llm_agent.process_prompt({"prompt": prompt})
                    response_text = response
                elif hasattr(self.llm_agent, '__call__'):
                    logger.info("Using __call__ method to call CoreLLM")
                    response = self.llm_agent(prompt)
                    response_text = response
                else:
                    logger.error("Cannot find a suitable method to call CoreLLM")
                    return self._get_fallback_analysis(query, "Cannot find a suitable method to call CoreLLM")
                
                # Handle different response formats
                if isinstance(response_text, dict) and "response" in response_text:
                    response_text = response_text["response"]
                elif not isinstance(response_text, str):
                    response_text = str(response_text)
                    
                logger.info(f"CoreLLM Response: {response_text[:100]}...")
            except Exception as e:
                logger.error(f"Error calling CoreLLM: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                return self._get_fallback_analysis(query, f"Error calling CoreLLM: {str(e)}")
            
            # Parse the JSON response
            try:
                # Find JSON in the response - sometimes the model includes other text
                json_match = re.search(r'(\{[\s\S]*\})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                    analysis = json.loads(json_str)
                    
                    # Validate the analysis structure
                    if self._validate_analysis(analysis):
                        analysis["ai_source"] = "real_core_llm"  # Mark this as coming from the real LLM
                        return analysis
                    else:
                        logger.warning(f"Invalid analysis structure: {analysis}")
                        return self._get_fallback_analysis(query, "Invalid analysis structure")
                else:
                    logger.warning(f"No JSON found in response: {response_text}")
                    return self._get_fallback_analysis(query, "No JSON found in response")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from CoreLLM response: {str(e)}")
                logger.debug(f"Raw response: {response_text}")
                return self._get_fallback_analysis(query, f"JSON parse error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error analyzing query with CoreLLM: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return self._get_fallback_analysis(query, str(e))
    
    async def _run_with_timeout(self, prompt: str) -> Dict[str, Any]:
        """
        Run the LLM with a timeout to prevent hanging.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response
        """
        loop = asyncio.get_event_loop()
        try:
            logger.info("Starting LLM generation with timeout")
            # Record message size to help with diagnostics
            token_estimate = len(prompt.split()) # Rough estimate
            logger.info(f"Prompt length estimate: {token_estimate} tokens")
            
            # Try to use a more detailed error handler
            # Create a future to run the LLM call in a separate thread
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    lambda: self.llm_agent.generate_response(
                        prompt=prompt,
                        temperature=0.1,
                        max_tokens=1024
                    )
                ),
                timeout=300.0  # Increased timeout to 300 seconds (5 minutes)
            )
            
            logger.info("LLM generation completed successfully")
            return response
        except asyncio.TimeoutError:
            logger.error("LLM generation timed out")
            raise
        except Exception as e:
            logger.error(f"Error during LLM generation: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            # Include stack trace for debugging
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        Validate that the analysis structure is correct.
        
        Args:
            analysis: The analysis to validate
            
        Returns:
            True if the analysis is valid, False otherwise
        """
        required_keys = ["operations", "concepts", "required_agents", "complexity"]
        for key in required_keys:
            if key not in analysis:
                return False
        
        # Validate types
        if not isinstance(analysis.get("operations", []), list):
            return False
        if not isinstance(analysis.get("concepts", []), list):
            return False
        if not isinstance(analysis.get("required_agents", []), list):
            return False
        if not isinstance(analysis.get("complexity", ""), str):
            return False
        
        return True
    
    def _get_fallback_analysis(self, query: str, error: str) -> Dict[str, Any]:
        """
        Get a fallback analysis for when the CoreLLM fails.
        
        Args:
            query: The original query
            error: The error message
            
        Returns:
            A basic analysis structure
        """
        query_lower = query.lower()
        
        # Log the fallback reason
        logger.warning(f"Using fallback analysis mechanism. Reason: {error}")
        
        # Basic rule-based fallback analysis
        operations = []
        concepts = []
        required_agents = ["core_llm_agent"]  # Always include the core LLM agent
        complexity = "simple"
        sub_problems = []
        
        # Detect equation solving
        if re.search(r'solve|equation|find.*solution|root', query_lower):
            operations.append("equation_solving")
            required_agents.append("math_computation_agent")
            
            if re.search(r'quadratic|x\^2|x²', query_lower):
                concepts.append("quadratic_equations")
            elif re.search(r'linear', query_lower):
                concepts.append("linear_equations")
        
        # Detect calculus operations
        if re.search(r'derivative|differentiate|slope|rate of change', query_lower):
            operations.append("differentiation")
            concepts.append("calculus")
            required_agents.append("math_computation_agent")
        
        if re.search(r'integral|integrate|area under', query_lower):
            operations.append("integration")
            concepts.append("calculus")
            required_agents.append("math_computation_agent")
        
        # Detect visualization needs
        if re.search(r'plot|graph|visualize|draw', query_lower):
            operations.append("plotting")
            required_agents.append("visualization_agent")
            
            if re.search(r'3d|three dimensional|surface', query_lower):
                operations.append("3d_plotting")
            
        # Create the fallback analysis
        fallback_analysis = {
            "operations": operations,
            "concepts": concepts,
            "required_agents": required_agents,
            "complexity": complexity,
            "sub_problems": sub_problems,
            "routing": {
                "primary_agent": required_agents[0],
                "confidence": 0.5,
                "alternative_agents": required_agents
            },
            "ai_source": "fallback_rules",
            "fallback": True,
            "error": error
        }
        
        return fallback_analysis


# Create a single agent instance to be reused - now using the real CoreLLM
try:
    ai_agent = RealAIAnalysisAgent()
    logger.info("Successfully initialized RealAIAnalysisAgent with CoreLLM")
except Exception as e:
    logger.error(f"Failed to initialize RealAIAnalysisAgent: {str(e)}")
    # Fall back to the simulated agent if needed
    from .query_analysis import SimpleAIAnalysisAgent
    ai_agent = SimpleAIAnalysisAgent()
    logger.warning("Falling back to SimpleAIAnalysisAgent")


class QueryAnalysisRequest(BaseModel):
    """Request model for AI query analysis."""
    query: str
    context_id: Optional[str] = None
    conversation_id: Optional[str] = None


class QueryAnalysisResponse(BaseModel):
    """Response model for AI query analysis."""
    success: bool
    query: str
    analysis: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    steps: Optional[List[str]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


@router.post("/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(request: QueryAnalysisRequest):
    """
    Analyze a mathematical query using the real CoreLLM.
    
    This endpoint uses the CoreLLM to:
    1. Analyze the query to determine mathematical operations and concepts
    2. Identify the appropriate specialized agents to handle the query
    3. Determine the complexity and structure of the query
    
    Returns detailed AI-generated analysis of the mathematical query.
    """
    try:
        start_time = time.time()
        
        # Log that we're using the real CoreLLM
        logger.info(f"Analyzing query with real CoreLLM: {request.query[:50]}...")
        
        # Analyze the query using our real AI agent
        try:
            analysis = await ai_agent.analyze_query(request.query)
        except Exception as e:
            logger.error(f"Error in AI analysis agent: {str(e)}")
            analysis = {
                "operations": [],
                "concepts": [],
                "required_agents": ["core_llm_agent"],
                "complexity": "simple",
                "sub_problems": [],
                "routing": {
                    "primary_agent": "core_llm_agent",
                    "confidence": 0.5,
                    "alternative_agents": ["core_llm_agent"]
                },
                "ai_source": "error_fallback",
                "fallback": True,
                "error": f"Error in AI analysis: {str(e)}"
            }
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Extract known special cases for answer and steps
        answer = None
        steps = None
        
        
        return QueryAnalysisResponse(
            success=True,
            query=request.query,
            analysis=analysis,
            answer=answer,
            steps=steps,
            execution_time=round(execution_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error in AI analysis endpoint: {str(e)}")
        return QueryAnalysisResponse(
            success=False,
            query=request.query,
            error=f"Failed to analyze query: {str(e)}"
        ) 
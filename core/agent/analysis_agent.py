"""
AI-based Analysis Agent for Mathematical Queries.

This module provides an AI-powered agent for analyzing and categorizing
mathematical queries using the loaded LLM.
"""

import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Union
from core.agent.llm_agent import CoreLLMAgent
# Remove non-existent dependencies
# from core.inference.inference_base import InferenceBase
# from core.utils.prompt_templates import load_prompt_template

logger = logging.getLogger(__name__)

class MathAnalysisAgent:
    """
    AI-powered agent for analyzing mathematical queries.
    
    This agent uses the loaded LLM to analyze mathematical queries and determine
    the operations, concepts, required agents, and complexity.
    """
    
    def __init__(self, llm_agent: Optional[CoreLLMAgent] = None):
        """
        Initialize the math analysis agent.
        
        Args:
            llm_agent: The CoreLLMAgent instance to use for inference
        """
        self.llm_agent = llm_agent
        
        # Analysis prompt template
        self.analysis_prompt_template = """
        [INST]
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
        [/INST]
        """
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a mathematical query using the LLM.
        
        Args:
            query: The mathematical query to analyze
            
        Returns:
            A dictionary containing the analysis results
        """
        try:
            if not self.llm_agent:
                logger.error("LLM agent not available for analysis")
                return self._get_fallback_analysis(query, "LLM agent not available")
            
            # Format the prompt with the query
            prompt = self.analysis_prompt_template.format(query=query)
            
            # Get response from LLM
            response = await self._get_llm_response(prompt)
            
            # Parse the JSON response
            try:
                # Find JSON in the response - sometimes the model includes other text
                json_match = re.search(r'(\{[\s\S]*\})', response)
                if json_match:
                    json_str = json_match.group(1)
                    analysis = json.loads(json_str)
                    
                    # Validate the analysis structure
                    if not self._validate_analysis(analysis):
                        logger.warning(f"Invalid analysis structure: {analysis}")
                        return self._get_fallback_analysis(query, "Invalid analysis structure")
                    
                    return analysis
                else:
                    logger.warning(f"No JSON found in response: {response}")
                    return self._get_fallback_analysis(query, "No JSON found in response")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from LLM response: {str(e)}")
                logger.debug(f"Raw response: {response}")
                return self._get_fallback_analysis(query, f"JSON parse error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error analyzing query with LLM: {str(e)}")
            return self._get_fallback_analysis(query, str(e))
    
    async def _get_llm_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        try:
            # Call the LLM - adapt this to match the actual interface of CoreLLMAgent
            # Try different methods that might be available
            if hasattr(self.llm_agent, 'generate_text'):
                response = self.llm_agent.generate_text(prompt, temperature=0.1, max_tokens=1024)
                return response
            elif hasattr(self.llm_agent, 'generate'):
                response = self.llm_agent.generate(prompt, temperature=0.1, max_tokens=1024)
                return response
            elif hasattr(self.llm_agent, 'process_text'):
                response = self.llm_agent.process_text(prompt)
                return response
            else:
                # Generic fallback - try to call the agent directly
                response = self.llm_agent(prompt)
                if isinstance(response, dict) and 'response' in response:
                    return response['response']
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
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
        Get a fallback analysis for when the LLM fails.
        
        Args:
            query: The original query
            error: The error message
            
        Returns:
            A basic analysis structure
        """
        query_lower = query.lower()
        
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
        
        # Determine primary agent
        primary_agent = "core_llm_agent"
        if "math_computation_agent" in required_agents:
            primary_agent = "math_computation_agent"
        elif "visualization_agent" in required_agents:
            primary_agent = "visualization_agent"
        
        # Add fallback info
        return {
            "operations": operations,
            "concepts": concepts,
            "required_agents": list(set(required_agents)),
            "complexity": complexity,
            "sub_problems": sub_problems,
            "routing": {
                "primary_agent": primary_agent,
                "confidence": 0.5,  # Low confidence for fallback
                "alternative_agents": ["core_llm_agent"]
            },
            "fallback": True,
            "error": error
        } 
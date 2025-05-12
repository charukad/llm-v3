"""
Integrated Response Workflow

This module defines the workflow for generating integrated responses by coordinating
the outputs from multiple specialized agents (Core LLM, Mathematical Computation,
Visualization, and Search) into a cohesive, well-structured response.
"""

import logging
import uuid
import datetime
from typing import Dict, List, Optional, Union, Any

from orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus
from orchestration.agents.registry import AgentRegistry
from math_processing.formatting.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class IntegratedResponseWorkflow:
    """
    Coordinates the generation of integrated mathematical responses.
    
    This workflow orchestrates the interaction between specialized agents:
    1. Analyzes the query to determine required computation and visualization
    2. Dispatches appropriate requests to specialized agents
    3. Collects and integrates results into a coherent response
    4. Formats the response according to user preferences
    """
    
    def __init__(self, message_bus: Optional[RabbitMQBus] = None, agent_registry: Optional[AgentRegistry] = None):
        """
        Initialize the workflow.
        
        Args:
            message_bus: Message bus for agent communication
            agent_registry: Registry of available agents
        """
        self.message_bus = message_bus or RabbitMQBus()
        self.agent_registry = agent_registry or AgentRegistry()
        self.response_formatter = ResponseFormatter()
        
    async def execute(self, 
                    query: str, 
                    context: Optional[Dict[str, Any]] = None,
                    format_type: str = "default",
                    complexity_level: str = "auto",
                    include_visualizations: bool = True,
                    include_step_by_step: bool = True,
                    include_search: bool = True) -> Dict[str, Any]:
        """
        Execute the integrated response workflow.
        
        Args:
            query: User's mathematical query
            context: Additional context information
            format_type: Response format type
            complexity_level: Complexity level for response
            include_visualizations: Whether to include visualizations
            include_step_by_step: Whether to include step-by-step solutions
            include_search: Whether to include external information from search
            
        Returns:
            Integrated response data
        """
        workflow_id = str(uuid.uuid4())
        logger.info(f"Starting integrated response workflow {workflow_id} for query: {query}")
        
        # Step 1: Initialize workflow data
        workflow_data = {
            "workflow_id": workflow_id,
            "query": query,
            "context": context or {},
            "start_time": datetime.datetime.now().isoformat(),
            "format_type": format_type,
            "complexity_level": complexity_level,
            "include_visualizations": include_visualizations,
            "include_step_by_step": include_step_by_step,
            "include_search": include_search
        }
        
        try:
            # Step 2: Analyze query with Core LLM Agent
            query_analysis = await self._analyze_query(query, context)
            workflow_data["query_analysis"] = query_analysis
            
            # Step 3: Perform mathematical computation if needed
            if query_analysis.get("requires_computation", False):
                computation_result = await self._perform_computation(
                    query, 
                    query_analysis, 
                    include_step_by_step
                )
                workflow_data["computation_result"] = computation_result
            
            # Step 4: Generate natural language explanation
            explanation = await self._generate_explanation(
                query, 
                query_analysis,
                workflow_data.get("computation_result")
            )
            workflow_data["explanation"] = explanation
            
            # Step 5: Generate visualizations if appropriate
            if include_visualizations and query_analysis.get("visualization_type"):
                visualizations = await self._generate_visualizations(
                    query,
                    query_analysis,
                    workflow_data.get("computation_result")
                )
                workflow_data["visualizations"] = visualizations
            
            # Step 6: Perform search for external information if needed
            if include_search and query_analysis.get("requires_search", False):
                search_results = await self._perform_search(
                    query,
                    query_analysis
                )
                workflow_data["search_results"] = search_results
            
            # Step 7: Format the integrated response
            integrated_response = self._format_integrated_response(
                workflow_data,
                format_type,
                complexity_level
            )
            
            # Log successful completion
            logger.info(f"Completed integrated response workflow {workflow_id}")
            
            return integrated_response
            
        except Exception as e:
            logger.error(f"Error in integrated response workflow {workflow_id}: {str(e)}")
            # Create error response
            error_response = {
                "workflow_id": workflow_id,
                "error": str(e),
                "explanation": f"An error occurred while processing your query: {str(e)}",
                "format_type": format_type,
                "complexity_level": complexity_level
            }
            return error_response
    
    async def _analyze_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the mathematical query to determine required operations.
        
        Args:
            query: User's mathematical query
            context: Additional context information
            
        Returns:
            Analysis result with identified operations and requirements
        """
        logger.info(f"Analyzing query: {query}")
        
        # Find the Core LLM Agent
        core_llm_agents = self.agent_registry.find_agent_by_capability("analyze_math_query")
        if not core_llm_agents:
            raise ValueError("No Core LLM Agent available for query analysis")
            
        agent_id = core_llm_agents[0]
        
        # Prepare the request
        request = {
            "query": query,
            "context": context
        }
        
        # Send the request to the Core LLM Agent
        response = await self.message_bus.send_request_async(
            recipient=agent_id,
            message_body=request,
            message_type="analyze_math_query"
        )
        
        if not response or "analysis" not in response:
            raise ValueError("Failed to analyze query")
            
        return response["analysis"]
    
    async def _perform_computation(self, 
                                 query: str, 
                                 query_analysis: Dict[str, Any],
                                 include_step_by_step: bool) -> Dict[str, Any]:
        """
        Perform mathematical computation based on query analysis.
        
        Args:
            query: User's mathematical query
            query_analysis: Result of query analysis
            include_step_by_step: Whether to include step-by-step solution
            
        Returns:
            Computation result
        """
        logger.info(f"Performing computation for query: {query}")
        
        # Extract computation details from query analysis
        operation = query_analysis.get("operation")
        expression = query_analysis.get("expression")
        domain = query_analysis.get("domain")
        
        if not operation or not expression:
            raise ValueError("Missing operation or expression for computation")
            
        # Find the Math Computation Agent
        math_agents = self.agent_registry.find_agent_by_capability("compute_math")
        if not math_agents:
            raise ValueError("No Math Computation Agent available")
            
        agent_id = math_agents[0]
        
        # Prepare the request
        request = {
            "operation": operation,
            "expression": expression,
            "domain": domain,
            "include_steps": include_step_by_step
        }
        
        # Send the request to the Math Computation Agent
        response = await self.message_bus.send_request_async(
            recipient=agent_id,
            message_body=request,
            message_type="compute_math"
        )
        
        if not response or "result" not in response:
            raise ValueError("Failed to perform computation")
            
        return response
    
    async def _generate_explanation(self, 
                                  query: str, 
                                  query_analysis: Dict[str, Any],
                                  computation_result: Optional[Dict[str, Any]]) -> str:
        """
        Generate a natural language explanation for the mathematical query.
        
        Args:
            query: User's mathematical query
            query_analysis: Result of query analysis
            computation_result: Result of mathematical computation
            
        Returns:
            Natural language explanation
        """
        logger.info(f"Generating explanation for query: {query}")
        
        # Find the Core LLM Agent
        core_llm_agents = self.agent_registry.find_agent_by_capability("generate_math_explanation")
        if not core_llm_agents:
            raise ValueError("No Core LLM Agent available for explanation generation")
            
        agent_id = core_llm_agents[0]
        
        # Prepare the request
        request = {
            "query": query,
            "analysis": query_analysis,
            "computation_result": computation_result
        }
        
        # Send the request to the Core LLM Agent
        response = await self.message_bus.send_request_async(
            recipient=agent_id,
            message_body=request,
            message_type="generate_math_explanation"
        )
        
        if not response or "explanation" not in response:
            raise ValueError("Failed to generate explanation")
            
        return response["explanation"]
    
    async def _generate_visualizations(self, 
                                     query: str, 
                                     query_analysis: Dict[str, Any],
                                     computation_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate visualizations based on query analysis and computation results.
        
        Args:
            query: User's mathematical query
            query_analysis: Result of query analysis
            computation_result: Result of mathematical computation
            
        Returns:
            List of visualization metadata
        """
        logger.info(f"Generating visualizations for query: {query}")
        
        # Extract visualization details from query analysis
        visualization_type = query_analysis.get("visualization_type")
        if not visualization_type:
            return []
            
        # Find the Visualization Agent
        viz_agents = self.agent_registry.find_agent_by_capability("generate_visualization")
        if not viz_agents:
            logger.warning("No Visualization Agent available, skipping visualization")
            return []
            
        agent_id = viz_agents[0]
        
        # Prepare the request
        request = {
            "query": query,
            "visualization_type": visualization_type,
            "analysis": query_analysis,
            "computation_result": computation_result
        }
        
        # Send the request to the Visualization Agent
        response = await self.message_bus.send_request_async(
            recipient=agent_id,
            message_body=request,
            message_type="generate_visualization"
        )
        
        if not response or "visualizations" not in response:
            logger.warning("Failed to generate visualizations")
            return []
            
        return response["visualizations"]
    
    async def _perform_search(self, 
                            query: str, 
                            query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform external search for additional information.
        
        Args:
            query: User's mathematical query
            query_analysis: Result of query analysis
            
        Returns:
            Search results
        """
        logger.info(f"Performing search for query: {query}")
        
        # Find the Search Agent
        search_agents = self.agent_registry.find_agent_by_capability("search")
        if not search_agents:
            logger.warning("No Search Agent available, skipping search")
            return []
            
        agent_id = search_agents[0]
        
        # Extract search terms from query analysis
        search_terms = query_analysis.get("search_terms", [query])
        domain = query_analysis.get("domain")
        
        # Prepare the request
        request = {
            "search_terms": search_terms,
            "domain": domain,
            "limit": 3  # Limit to top 3 results
        }
        
        # Send the request to the Search Agent
        response = await self.message_bus.send_request_async(
            recipient=agent_id,
            message_body=request,
            message_type="search"
        )
        
        if not response or "results" not in response:
            logger.warning("Failed to perform search")
            return []
            
        return response["results"]
    
    def _format_integrated_response(self, 
                                   workflow_data: Dict[str, Any],
                                   format_type: str,
                                   complexity_level: str) -> Dict[str, Any]:
        """
        Format the integrated response using collected workflow data.
        
        Args:
            workflow_data: Collected data from workflow steps
            format_type: Response format type
            complexity_level: Complexity level for response
            
        Returns:
            Formatted integrated response
        """
        logger.info("Formatting integrated response")
        
        # Extract components for the response
        explanation = workflow_data.get("explanation", "")
        
        # Extract LaTeX expressions from computation result
        latex_expressions = []
        computation_result = workflow_data.get("computation_result", {})
        if computation_result:
            if "latex_result" in computation_result:
                latex_expressions.append(computation_result["latex_result"])
            elif "result" in computation_result:
                latex_expressions.append(str(computation_result["result"]))
        
        # Extract solution steps
        steps = computation_result.get("steps", [])
        
        # Extract visualizations
        visualizations = workflow_data.get("visualizations", [])
        
        # Extract citations from search results
        citations = []
        search_results = workflow_data.get("search_results", [])
        if search_results:
            for result in search_results:
                if "title" in result and "url" in result:
                    citations.append(f"{result.get('title')} - {result.get('url')}")
        
        # Prepare response data for formatting
        response_data = {
            "explanation": explanation,
            "latex_expressions": latex_expressions,
            "steps": steps,
            "visualizations": visualizations,
            "citations": citations,
            "domain": workflow_data.get("query_analysis", {}).get("domain")
        }
        
        # Format the response
        formatted_response = self.response_formatter.format_response(
            response_data=response_data,
            format_type=format_type,
            complexity_level=complexity_level,
            include_citations=(len(citations) > 0)
        )
        
        # Add workflow metadata
        formatted_response["workflow_id"] = workflow_data["workflow_id"]
        formatted_response["query"] = workflow_data["query"]
        formatted_response["processing_time"] = self._calculate_processing_time(
            workflow_data.get("start_time")
        )
        
        return formatted_response
    
    def _calculate_processing_time(self, start_time: Optional[str]) -> Optional[float]:
        """
        Calculate processing time in seconds.
        
        Args:
            start_time: ISO format start time
            
        Returns:
            Processing time in seconds or None if start_time is not provided
        """
        if not start_time:
            return None
            
        try:
            start_datetime = datetime.datetime.fromisoformat(start_time)
            end_datetime = datetime.datetime.now()
            return (end_datetime - start_datetime).total_seconds()
        except Exception as e:
            logger.error(f"Error calculating processing time: {str(e)}")
            return None

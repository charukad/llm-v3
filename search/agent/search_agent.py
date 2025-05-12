"""
Search Agent for retrieving mathematical information from external sources.

This module implements the Search Agent, which supplements the system's
knowledge by retrieving and integrating information from reliable external sources.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import threading
import queue

from search.api.google_search import GoogleSearchAPI, ArxivSearchAPI, WolframAlphaAPI
from search.query.query_generator import MathSearchQueryGenerator, QueryOptimizer
from search.processing.math_content_parser import MathContentExtractor, MathCredibilityEvaluator, RelevanceScoringEngine
from orchestration.message_bus.rabbitmq_wrapper import RabbitMQBus

logger = logging.getLogger(__name__)

class SearchAgent:
    """
    Agent for retrieving mathematical information from external sources.
    
    This agent handles search queries, retrieves information from various sources,
    processes and evaluates the results, and formats them for integration with
    the system's responses.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Search Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize search APIs
        self.google_search = GoogleSearchAPI(
            api_key=self.config.get("google_api_key"),
            cx=self.config.get("google_cx")
        )
        
        self.arxiv_search = ArxivSearchAPI()
        
        self.wolfram_alpha = WolframAlphaAPI(
            app_id=self.config.get("wolfram_app_id")
        )
        
        # Initialize query generation components
        self.query_generator = MathSearchQueryGenerator()
        self.query_optimizer = QueryOptimizer(self.query_generator)
        
        # Initialize content processing components
        self.content_extractor = MathContentExtractor()
        self.credibility_evaluator = MathCredibilityEvaluator()
        self.relevance_scorer = RelevanceScoringEngine()
        
        # Initialize message bus connection
        self.message_bus = None
        if self.config.get("message_bus_host"):
            self.message_bus = RabbitMQBus(
                host=self.config.get("message_bus_host"),
                port=self.config.get("message_bus_port", 5672)
            )
        
        # Register agent capabilities
        self._register_capabilities()
        
        # Search result cache
        self.result_cache = {}
        
        # Threading setup for parallel searches
        self.search_threads = {}
        self.results_queue = queue.Queue()
    
    def _register_capabilities(self):
        """Register agent capabilities with the orchestration system."""
        if not self.message_bus:
            logger.warning("Message bus not configured. Cannot register capabilities.")
            return
        
        capabilities = [
            "search_internet",
            "verify_information",
            "retrieve_mathematical_content",
            "supplement_knowledge"
        ]
        
        agent_info = {
            "agent_id": "search_agent",
            "agent_type": "search",
            "capabilities": capabilities,
            "status": "active"
        }
        
        try:
            self.message_bus.send_message(
                recipient="orchestration_manager",
                message_body={"agent_info": agent_info},
                message_type="agent_registration"
            )
            logger.info("Successfully registered Search Agent capabilities")
        except Exception as e:
            logger.error(f"Failed to register capabilities: {str(e)}")
    
    def search(self, query: str, domain: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a search for mathematical information.
        
        Args:
            query: Mathematical query or question
            domain: Optional mathematical domain for context
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results and processed content
        """
        # Check cache first
        cache_key = f"{query}_{domain}_{max_results}"
        if cache_key in self.result_cache:
            logger.info(f"Returning cached results for query: {query}")
            cached_result = self.result_cache[cache_key]
            
            # Add cache information
            cached_result["from_cache"] = True
            return cached_result
        
        logger.info(f"Performing search for query: {query}")
        
        # Generate optimized search queries
        search_queries = self.query_generator.generate_queries(query, domain)
        
        # Perform searches using multiple queries in parallel
        search_results = self._parallel_search(search_queries, max_results)
        
        # If no results, try to optimize queries and search again
        if not search_results.get("success", False) or not search_results.get("items", []):
            logger.info("No results found, optimizing queries")
            optimized_queries = self.query_optimizer._generate_fallback_queries(query)
            search_results = self._parallel_search(optimized_queries, max_results)
        
        # Extract content from search results
        extracted_results = self.content_extractor.extract_from_search_results(search_results, max_results=3)
        
        # Score results for relevance
        scored_results = self.relevance_scorer.rank_search_results(extracted_results, query)
        
        # Evaluate credibility of sources
        credibility_results = self._evaluate_credibility(scored_results)
        
        # Format results for integration
        formatted_results = self._format_for_integration(credibility_results, query)
        
        # Cache the results
        self.result_cache[cache_key] = formatted_results
        
        return formatted_results
    
    def _parallel_search(self, queries: List[str], max_results_per_query: int = 3) -> Dict[str, Any]:
        """
        Perform parallel searches with multiple queries.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            
        Returns:
            Combined search results
        """
        # Clear previous results
        self.results_queue = queue.Queue()
        
        # Start a thread for each query
        for i, query in enumerate(queries[:3]):  # Limit to top 3 queries
            thread = threading.Thread(
                target=self._search_thread,
                args=(query, max_results_per_query, i)
            )
            self.search_threads[i] = thread
            thread.start()
        
        # Wait for all threads to complete or timeout
        timeout = 15  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout and any(thread.is_alive() for thread in self.search_threads.values()):
            time.sleep(0.1)
        
        # Collect results
        all_results = []
        while not self.results_queue.empty():
            result = self.results_queue.get()
            all_results.append(result)
        
        # Sort by success and number of items
        all_results.sort(key=lambda x: (not x.get("success", False), -len(x.get("items", []))))
        
        # Use the best result, or combine if needed
        if not all_results:
            return {
                "success": False,
                "error": "No search results available",
                "items": []
            }
        
        best_result = all_results[0]
        
        # If best result has items, return it
        if best_result.get("success", False) and best_result.get("items", []):
            return best_result
        
        # Otherwise, combine results from all queries
        combined_items = []
        for result in all_results:
            if result.get("success", False):
                for item in result.get("items", []):
                    if not any(i.get("link") == item.get("link") for i in combined_items):
                        combined_items.append(item)
        
        return {
            "success": len(combined_items) > 0,
            "items": combined_items[:max_results_per_query],
            "query": queries[0]  # Use first query as reference
        }
    
    def _search_thread(self, query: str, max_results: int, thread_id: int):
        """
        Thread function for performing a search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            thread_id: Thread identifier
        """
        try:
            result = self.google_search.search(query, max_results)
            self.results_queue.put(result)
        except Exception as e:
            logger.error(f"Error in search thread {thread_id}: {str(e)}")
            self.results_queue.put({
                "success": False,
                "error": str(e),
                "items": []
            })
    
    def _evaluate_credibility(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate credibility of search results.
        
        Args:
            search_results: Search results with extracted content
            
        Returns:
            Results with added credibility information
        """
        # Check if results are available
        if not search_results.get("success", False) or not search_results.get("individual_results", []):
            return search_results
        
        # Add credibility evaluation to individual results
        for result in search_results.get("scored_results", []):
            content = result["content"]
            credibility = self.credibility_evaluator.evaluate_content(content)
            result["credibility"] = credibility
        
        # Calculate overall credibility score for aggregated content
        overall_credibility = 0.0
        total_weight = 0.0
        
        for result in search_results.get("scored_results", []):
            # Weight by both relevance and credibility
            weight = result["relevance_score"] * result["credibility"]["score"]
            overall_credibility += weight * result["credibility"]["score"]
            total_weight += weight
        
        if total_weight > 0:
            overall_credibility /= total_weight
        
        # Add overall credibility to results
        search_results["overall_credibility"] = overall_credibility
        
        return search_results
    
    def _format_for_integration(self, search_results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Format search results for integration with the system.
        
        Args:
            search_results: Search results with extracted content and credibility
            original_query: Original query string
            
        Returns:
            Formatted results for integration
        """
        # Check if results are available
        if not search_results.get("success", False):
            return {
                "success": False,
                "message": "No relevant information found",
                "query": original_query,
                "formatted_content": None
            }
        
        # Get aggregated content
        aggregated = search_results.get("aggregated_content", {})
        
        # Format content for integration
        formatted_content = {
            "context": self._generate_context_text(search_results, original_query),
            "latex_expressions": aggregated.get("latex_expressions", [])[:5],
            "theorems": aggregated.get("theorems", [])[:2],
            "definitions": aggregated.get("definitions", [])[:2],
            "examples": aggregated.get("examples", [])[:2],
            "sources": self._format_sources(search_results),
            "query": original_query,
            "credibility": search_results.get("overall_credibility", 0.0)
        }
        
        return {
            "success": True,
            "message": "Found relevant information",
            "query": original_query,
            "formatted_content": formatted_content
        }
    
    def _generate_context_text(self, search_results: Dict[str, Any], original_query: str) -> str:
        """
        Generate context text summarizing the search results.
        
        Args:
            search_results: Search results with extracted content
            original_query: Original query string
            
        Returns:
            Context text for LLM
        """
        # Get top scored results
        top_results = []
        for result in search_results.get("scored_results", []):
            if result["relevance_score"] > 0.4 and result["credibility"]["score"] > 0.3:
                top_results.append(result)
        
        if not top_results:
            return f"No high-confidence information was found for the query: {original_query}"
        
        # Extract relevant text from top results
        context_items = []
        
        # Add highest scoring definitions
        definitions = []
        for result in top_results:
            content = result["content"]
            for definition in content.get("definitions", [])[:1]:  # Take at most one from each
                if definition.get("content") and len(definition.get("content")) > 20:
                    definitions.append({
                        "content": definition.get("content"),
                        "score": result["relevance_score"] * result["credibility"]["score"]
                    })
        
        # Sort by score and take top 2
        definitions.sort(key=lambda x: x["score"], reverse=True)
        for definition in definitions[:2]:
            context_items.append(f"DEFINITION: {definition['content']}")
        
        # Add highest scoring theorems
        theorems = []
        for result in top_results:
            content = result["content"]
            for theorem in content.get("theorems", [])[:1]:  # Take at most one from each
                if theorem.get("content") and len(theorem.get("content")) > 20:
                    theorems.append({
                        "title": theorem.get("title", ""),
                        "content": theorem.get("content"),
                        "score": result["relevance_score"] * result["credibility"]["score"]
                    })
        
        # Sort by score and take top 2
        theorems.sort(key=lambda x: x["score"], reverse=True)
        for theorem in theorems[:2]:
            if theorem.get("title"):
                context_items.append(f"THEOREM ({theorem['title']}): {theorem['content']}")
            else:
                context_items.append(f"THEOREM: {theorem['content']}")
        
        # Add example
        examples = []
        for result in top_results:
            content = result["content"]
            for example in content.get("examples", [])[:1]:  # Take at most one from each
                if example and len(example) > 20:
                    examples.append({
                        "content": example,
                        "score": result["relevance_score"] * result["credibility"]["score"]
                    })
        
        # Sort by score and take top 1
        examples.sort(key=lambda x: x["score"], reverse=True)
        for example in examples[:1]:
            context_items.append(f"EXAMPLE: {example['content']}")
        
        # Combine context items
        context_text = "\n\n".join(context_items)
        
        # If we have no specific content items, extract from main text
        if not context_items:
            for result in top_results[:1]:
                content = result["content"]
                main_text = content.get("main_text", "")
                
                # Extract a reasonable excerpt
                if main_text and len(main_text) > 50:
                    # Limit to 1000 characters
                    excerpt = main_text[:1000] + ("..." if len(main_text) > 1000 else "")
                    context_text = f"RELEVANT INFORMATION: {excerpt}"
        
        # Add introduction
        intro = f"The following information was found from reliable sources regarding: {original_query}\n\n"
        
        return intro + context_text
    
    def _format_sources(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format sources for citation.
        
        Args:
            search_results: Search results with extracted content
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        # Get sources from aggregated content
        for source in search_results.get("aggregated_content", {}).get("sources", []):
            sources.append({
                "url": source.get("url", ""),
                "title": source.get("title", "Unknown Source")
            })
        
        # Add sources from top scored results if needed
        if len(sources) < 2:
            for result in search_results.get("scored_results", [])[:3]:
                content = result["content"]
                url = content.get("source", "")
                title = content.get("title", "Unknown Source")
                
                # Check if already included
                if not any(s["url"] == url for s in sources):
                    sources.append({
                        "url": url,
                        "title": title
                    })
                
                # Limit to 3 sources
                if len(sources) >= 3:
                    break
        
        return sources
    
    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming messages from the orchestration system.
        
        Args:
            message: Message dictionary
            
        Returns:
            Response message
        """
        message_type = message.get("header", {}).get("message_type", "")
        body = message.get("body", {})
        
        if message_type == "search_request":
            query = body.get("query", "")
            domain = body.get("domain")
            max_results = body.get("max_results", 5)
            
            # Perform search
            result = self.search(query, domain, max_results)
            
            return {
                "header": {
                    "message_id": f"response_{message.get('header', {}).get('message_id', '')}",
                    "timestamp": time.time(),
                    "message_type": "search_response",
                    "sender": "search_agent",
                    "recipient": message.get("header", {}).get("sender", "")
                },
                "body": result
            }
        
        elif message_type == "verification_request":
            information = body.get("information", "")
            
            # Verify information using Wolfram Alpha
            result = self.wolfram_alpha.verify_computation(information)
            
            return {
                "header": {
                    "message_id": f"response_{message.get('header', {}).get('message_id', '')}",
                    "timestamp": time.time(),
                    "message_type": "verification_response",
                    "sender": "search_agent",
                    "recipient": message.get("header", {}).get("sender", "")
                },
                "body": result
            }
        
        elif message_type == "academic_search_request":
            query = body.get("query", "")
            max_results = body.get("max_results", 5)
            
            # Perform arXiv search
            result = self.arxiv_search.search(query, max_results)
            
            return {
                "header": {
                    "message_id": f"response_{message.get('header', {}).get('message_id', '')}",
                    "timestamp": time.time(),
                    "message_type": "academic_search_response",
                    "sender": "search_agent",
                    "recipient": message.get("header", {}).get("sender", "")
                },
                "body": result
            }
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return {
                "header": {
                    "message_id": f"response_{message.get('header', {}).get('message_id', '')}",
                    "timestamp": time.time(),
                    "message_type": "error_response",
                    "sender": "search_agent",
                    "recipient": message.get("header", {}).get("sender", "")
                },
                "body": {
                    "success": False,
                    "error": f"Unknown message type: {message_type}"
                }
            }
    
    def start_message_handling(self):
        """Start handling messages from the message bus."""
        if not self.message_bus:
            logger.warning("Message bus not configured. Cannot start message handling.")
            return
        
        try:
            # Set up queue for receiving messages
            self.message_bus.setup_consumer("search_agent", self._process_message)
            logger.info("Started message handling for Search Agent")
        except Exception as e:
            logger.error(f"Failed to start message handling: {str(e)}")
    
    def _process_message(self, message: Dict[str, Any]):
        """
        Process an incoming message and send response.
        
        Args:
            message: Message dictionary
        """
        try:
            response = self.handle_message(message)
            
            # Send response
            if self.message_bus and response:
                recipient = response.get("header", {}).get("recipient", "")
                if recipient:
                    self.message_bus.send_message(
                        recipient=recipient,
                        message_body=response.get("body", {}),
                        message_type=response.get("header", {}).get("message_type", "")
                    )
        except Exception as e:
            logger.exception(f"Error processing message: {str(e)}")
            
            # Send error response
            if self.message_bus:
                self.message_bus.send_message(
                    recipient=message.get("header", {}).get("sender", ""),
                    message_body={
                        "success": False,
                        "error": str(e),
                        "original_message_id": message.get("header", {}).get("message_id", "")
                    },
                    message_type="error_response"
                )
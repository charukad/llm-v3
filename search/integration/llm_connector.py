"""
Integration of Search Agent with Core LLM.

This module provides functionality to integrate the Search Agent with
the Core LLM for enhanced mathematical responses.
"""

import logging
from typing import Dict, List, Any, Optional

from search.agent.search_agent import SearchAgent
from search.integration.context_integrator import SearchContextIntegrator
from search.integration.citation_generator import CitationGenerator

logger = logging.getLogger(__name__)

class LLMConnector:
    """
    Connects the Search Agent with the Core LLM.
    
    This class provides methods to enhance LLM prompts with search results
    and to incorporate search results into final responses.
    """
    
    def __init__(self, search_agent: SearchAgent):
        """
        Initialize the LLM Connector.
        
        Args:
            search_agent: The Search Agent instance
        """
        self.search_agent = search_agent
        self.context_integrator = SearchContextIntegrator()
        self.citation_generator = CitationGenerator()
    
    def enhance_prompt(self, prompt: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance a prompt with information from search results.
        
        Args:
            prompt: Original prompt for the LLM
            domain: Optional mathematical domain for context
            
        Returns:
            Dictionary containing enhanced prompt and search results
        """
        logger.info(f"Enhancing prompt with search results: {prompt[:100]}...")
        
        # Determine if we should search for this prompt
        if self._should_perform_search(prompt):
            # Perform search
            search_results = self.search_agent.search(prompt, domain)
            
            # Check if search was successful
            if search_results.get("success", False) and search_results.get("formatted_content"):
                # Generate context addition
                context_text = self.context_integrator.integrate_search_results(
                    search_results, prompt
                )
                
                # Add context to prompt
                if context_text:
                    enhanced_prompt = self._build_enhanced_prompt(prompt, context_text)
                    
                    return {
                        "original_prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "search_results": search_results,
                        "context_added": True
                    }
        
        # If search failed or wasn't needed, return original prompt
        return {
            "original_prompt": prompt,
            "enhanced_prompt": prompt,
            "search_results": None,
            "context_added": False
        }
    
    def enhance_response(self, llm_output: str, search_results: Dict[str, Any], 
                        add_citations: bool = True, citation_style: str = "standard") -> str:
        """
        Enhance LLM response with search information.
        
        Args:
            llm_output: Original output from the LLM
            search_results: Search results used in the prompt
            add_citations: Whether to add citations
            citation_style: Citation style to use
            
        Returns:
            Enhanced response
        """
        # If no search results, return original output
        if not search_results or not search_results.get("success", False):
            return llm_output
        
        # Format response with citations if requested
        if add_citations:
            enhanced_output = self.citation_generator.format_response_with_citations(
                llm_output, search_results, citation_style
            )
            return enhanced_output
        
        return llm_output
    
    def _should_perform_search(self, prompt: str) -> bool:
        """
        Determine if a search should be performed for a prompt.
        
        Args:
            prompt: Prompt to check
            
        Returns:
            True if search should be performed, False otherwise
        """
        # Check if prompt is a mathematical question
        prompt_lower = prompt.lower()
        
        # Look for question indicators
        has_question = "?" in prompt or any(w in prompt_lower for w in ["what", "who", "when", "where", "why", "how", "which", "find", "solve", "compute", "calculate"])
        
        # Look for mathematical terms
        math_terms = [
            "theorem", "proof", "lemma", "corollary", "equation", "formula", "theory",
            "integral", "derivative", "function", "matrix", "vector", "converge",
            "diverge", "infinity", "limit", "definite", "indefinite", "probability",
            "statistic", "distribution", "eigenvalue", "eigenvector", "topology",
            "algebra", "calculus", "geometry", "analysis", "differential", "series"
        ]
        
        has_math_terms = any(term in prompt_lower for term in math_terms)
        
        # Check for specialized query indicators
        specialized_indicators = [
            "named after", "discover", "history", "prove", "disprove", "conjecture",
            "unsolved", "recent", "discovery", "breakthrough", "research", "paper",
            "published", "author", "mathematician"
        ]
        
        is_specialized = any(indicator in prompt_lower for indicator in specialized_indicators)
        
        # Decision logic
        if is_specialized:
            # Always search for specialized queries
            return True
        elif has_question and has_math_terms:
            # Search for mathematical questions
            return True
        else:
            # Don't search for non-mathematical or non-question prompts
            return False
    
    def _build_enhanced_prompt(self, original_prompt: str, context_text: str) -> str:
        """
        Build an enhanced prompt with search context.
        
        Args:
            original_prompt: Original prompt
            context_text: Context text from search
            
        Returns:
            Enhanced prompt
        """
        # Build system context addition
        system_context = (
            "I'll provide you with some additional information from reliable "
            "mathematical sources that may be relevant to the query. "
            "Use this information to enhance your response when appropriate.\n\n"
            "ADDITIONAL INFORMATION:\n"
            f"{context_text}\n\n"
            "USER QUERY:\n"
            f"{original_prompt}"
        )
        
        return system_context
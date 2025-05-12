"""
Citation generation for information retrieved from external sources.

This module provides functionality to generate properly formatted citations
for information retrieved by the Search Agent, supporting various citation styles.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CitationGenerator:
    """
    Generates citations for search results.
    
    This class provides methods to generate citations in various styles for
    information retrieved from external sources.
    """
    
    def __init__(self):
        """Initialize the citation generator."""
        # Citation style templates
        self.citation_styles = {
            "apa": self._format_apa,
            "mla": self._format_mla,
            "chicago": self._format_chicago,
            "harvard": self._format_harvard,
            "ieee": self._format_ieee,
            "standard": self._format_standard,
            "minimal": self._format_minimal
        }
    
    def generate_citations(self, search_results: Dict[str, Any], style: str = "standard") -> List[str]:
        """
        Generate citations for search results.
        
        Args:
            search_results: Formatted search results
            style: Citation style (apa, mla, chicago, harvard, ieee, standard, minimal)
            
        Returns:
            List of citation strings
        """
        if not search_results.get("success", False) or not search_results.get("formatted_content"):
            return []
        
        formatted_content = search_results.get("formatted_content", {})
        
        if not formatted_content.get("sources"):
            return []
        
        # Select citation formatter
        formatter = self.citation_styles.get(style.lower(), self._format_standard)
        
        # Generate citations
        citations = []
        for source in formatted_content["sources"]:
            citation = formatter(source)
            if citation:
                citations.append(citation)
        
        return citations
    
    def format_response_with_citations(self, llm_output: str, search_results: Dict[str, Any], style: str = "standard") -> str:
        """
        Format LLM output with citations.
        
        Args:
            llm_output: Original LLM output
            search_results: Formatted search results
            style: Citation style
            
        Returns:
            Output with added citations
        """
        if not search_results.get("success", False) or not search_results.get("formatted_content"):
            return llm_output
        
        # Generate citations
        citations = self.generate_citations(search_results, style)
        
        if not citations:
            return llm_output
        
        # Add citation section
        citation_text = "\n\nReferences:\n"
        for i, citation in enumerate(citations, 1):
            citation_text += f"{i}. {citation}\n"
        
        return llm_output + citation_text
    
    def inject_inline_citations(self, llm_output: str, search_results: Dict[str, Any]) -> str:
        """
        Inject inline citations into LLM output.
        
        Args:
            llm_output: Original LLM output
            search_results: Formatted search results
            
        Returns:
            Output with added inline citations
        """
        if not search_results.get("success", False) or not search_results.get("formatted_content"):
            return llm_output
        
        formatted_content = search_results.get("formatted_content", {})
        
        if not formatted_content.get("sources"):
            return llm_output
        
        # Create source mapping for citation references
        sources = formatted_content["sources"]
        source_map = {i+1: source for i, source in enumerate(sources)}
        
        # Extract sentences from LLM output
        sentences = re.split(r'(?<=[.!?])\s+', llm_output)
        
        # Process each sentence to add citations
        processed_sentences = []
        for sentence in sentences:
            # Check if sentence contains mathematical content
            if self._contains_mathematical_content(sentence):
                # Find the most relevant source for this content
                source_idx = self._find_relevant_source(sentence, sources)
                
                # If a relevant source was found, add citation
                if source_idx is not None:
                    citation_idx = source_idx + 1
                    processed_sentences.append(f"{sentence} [{citation_idx}]")
                else:
                    processed_sentences.append(sentence)
            else:
                processed_sentences.append(sentence)
        
        # Combine processed sentences
        cited_output = " ".join(processed_sentences)
        
        # Add references section
        references = "\n\nReferences:\n"
        for idx, source in source_map.items():
            references += f"[{idx}] {self._format_standard(source)}\n"
        
        return cited_output + references
    
    def _contains_mathematical_content(self, text: str) -> bool:
        """
        Check if text contains mathematical content.
        
        Args:
            text: Input text
            
        Returns:
            True if text contains mathematical content, False otherwise
        """
        # Check for LaTeX expressions
        if "$" in text or "\\(" in text or "\\[" in text:
            return True
        
        # Check for mathematical terms
        math_terms = [
            "theorem", "proof", "lemma", "corollary", "equation", "formula",
            "integral", "derivative", "function", "matrix", "vector", "converge",
            "diverge", "infinity", "limit", "definite", "indefinite", "probability",
            "statistic", "distribution", "eigenvalue", "eigenvector"
        ]
        
        text_lower = text.lower()
        for term in math_terms:
            if term in text_lower:
                return True
        
        # Check for mathematical symbols
        math_symbols = ["+", "-", "*", "/", "=", "<", ">", "≤", "≥", "≠", "∈", "∉", "⊂", "⊃", "∩", "∪", "∫", "∑", "∏"]
        for symbol in math_symbols:
            if symbol in text:
                return True
        
        return False
    
    def _find_relevant_source(self, text: str, sources: List[Dict[str, str]]) -> Optional[int]:
        """
        Find the most relevant source for a text.
        
        Args:
            text: Input text
            sources: List of source dictionaries
            
        Returns:
            Index of the most relevant source, or None if no relevant source found
        """
        best_match = None
        best_score = 0.0
        
        for i, source in enumerate(sources):
            # Simple similarity based on common words
            title = source.get("title", "").lower()
            
            # Extract words from both texts
            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            title_words = set(re.findall(r'\b\w+\b', title))
            
            # Remove common words
            stop_words = {"the", "a", "an", "in", "on", "at", "of", "to", "for", "with", "by"}
            text_words = text_words.difference(stop_words)
            title_words = title_words.difference(stop_words)
            
            # Calculate overlap
            if not text_words or not title_words:
                continue
                
            intersection = len(text_words.intersection(title_words))
            score = intersection / len(title_words) if title_words else 0
            
            if score > best_score and score > 0.2:  # Threshold for relevance
                best_score = score
                best_match = i
        
        return best_match
    
    def _format_standard(self, source: Dict[str, str]) -> str:
        """
        Format source in standard format.
        
        Args:
            source: Source dictionary
            
        Returns:
            Formatted citation
        """
        url = source.get("url", "")
        title = source.get("title", "Unknown Source")
        
        if not url:
            return title
        
        return f"{title}. Retrieved from {url}"
    
    def _format_minimal(self, source: Dict[str, str]) -> str:
        """
        Format source in minimal format.
        
        Args:
            source: Source dictionary
            
        Returns:
            Formatted citation
        """
        url = source.get("url", "")
        
        if not url:
            return source.get("title", "Unknown Source")
        
        return url
    
    def _format_apa(self, source: Dict[str, str]) -> str:
        """
        Format source in APA style.
        
        Args:
            source: Source dictionary
            
        Returns:
            Formatted citation
        """
        url = source.get("url", "")
        title = source.get("title", "Unknown Source")
        date = source.get("accessed_date", datetime.now().strftime("%Y, %B %d"))
        
        if not url:
            return title
        
        # Extract domain for author if not provided
        author = source.get("author", "")
        if not author:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                author = domain_match.group(1)
            else:
                author = "n.d."
        
        return f"{author}. ({date}). {title}. Retrieved from {url}"
    
    def _format_mla(self, source: Dict[str, str]) -> str:
        """
        Format source in MLA style.
        
        Args:
            source: Source dictionary
            
        Returns:
            Formatted citation
        """
        url = source.get("url", "")
        title = source.get("title", "Unknown Source")
        date = source.get("accessed_date", datetime.now().strftime("%d %b. %Y"))
        
        if not url:
            return title
        
        # Extract domain for site name if not provided
        site_name = source.get("site_name", "")
        if not site_name:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                site_name = domain_match.group(1).capitalize()
            else:
                site_name = "Web"
        
        return f'"{title}." {site_name}, Accessed {date}, {url}.'
    
    def _format_chicago(self, source: Dict[str, str]) -> str:
        """
        Format source in Chicago style.
        
        Args:
            source: Source dictionary
            
        Returns:
            Formatted citation
        """
        url = source.get("url", "")
        title = source.get("title", "Unknown Source")
        date = source.get("accessed_date", datetime.now().strftime("%B %d, %Y"))
        
        if not url:
            return title
        
        # Extract domain for site name if not provided
        site_name = source.get("site_name", "")
        if not site_name:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                site_name = domain_match.group(1).capitalize()
            else:
                site_name = "Website"
        
        return f'"{title}." {site_name}. Accessed {date}. {url}.'
    
    def _format_harvard(self, source: Dict[str, str]) -> str:
        """
        Format source in Harvard style.
        
        Args:
            source: Source dictionary
            
        Returns:
            Formatted citation
        """
        url = source.get("url", "")
        title = source.get("title", "Unknown Source")
        year = source.get("year", datetime.now().year)
        date = source.get("accessed_date", datetime.now().strftime("%d %B %Y"))
        
        if not url:
            return title
        
        # Extract domain for author if not provided
        author = source.get("author", "")
        if not author:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                author = domain_match.group(1).capitalize()
            else:
                author = "Anon."
        
        return f"{author} ({year}). {title}. [online] Available at: {url} [Accessed {date}]."
    
    def _format_ieee(self, source: Dict[str, str]) -> str:
        """
        Format source in IEEE style.
        
        Args:
            source: Source dictionary
            
        Returns:
            Formatted citation
        """
        url = source.get("url", "")
        title = source.get("title", "Unknown Source")
        year = source.get("year", datetime.now().year)
        date = source.get("accessed_date", datetime.now().strftime("%b. %d, %Y"))
        
        if not url:
            return title
        
        # Extract domain for site name if not provided
        site_name = source.get("site_name", "")
        if not site_name:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                site_name = domain_match.group(1).capitalize()
            else:
                site_name = "Online"
        
        return f'"{title}," {site_name}. [Online]. Available: {url}. [Accessed: {date}].'
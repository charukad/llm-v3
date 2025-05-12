"""
Query generation for mathematical search.

This module generates effective search queries for mathematical content,
optimizing them for different search engines and mathematical domains.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from math_processing.expressions.latex_parser import parse_math_expression
from math_processing.classification.domain_classifier import classify_mathematical_domain

logger = logging.getLogger(__name__)

class MathSearchQueryGenerator:
    """
    Generator for mathematical search queries.
    
    This class creates optimized search queries for mathematical content,
    extracting key terms and adapting to different mathematical domains.
    """
    
    def __init__(self):
        """Initialize the mathematical search query generator."""
        # Domain-specific qualifiers to improve search relevance
        self.domain_qualifiers = {
            "algebra": ["algebra", "equation", "polynomial", "factorization"],
            "calculus": ["calculus", "differentiation", "integration", "limit"],
            "linear_algebra": ["linear algebra", "matrix", "vector space", "eigenvalue"],
            "statistics": ["statistics", "probability", "distribution", "hypothesis testing"],
            "geometry": ["geometry", "coordinates", "vector", "geometric proof"],
            "number_theory": ["number theory", "prime", "divisibility", "modular arithmetic"],
            "discrete_math": ["discrete mathematics", "combinatorics", "graph theory"]
        }
        
        # Keywords to extract from mathematical queries
        self.math_keywords = [
            "theorem", "proof", "lemma", "corollary", "equation", "formula",
            "integral", "derivative", "matrix", "vector", "function", "series",
            "convergence", "distribution", "eigenvalue", "eigenvector", "polynomial",
            "factorization", "group", "ring", "field", "limit", "continuity",
            "differential", "equation", "statistics", "probability", "algorithm"
        ]
    
    def generate_queries(self, mathematical_query: str, domain: Optional[str] = None) -> List[str]:
        """
        Generate appropriate search queries for a mathematical question.
        
        Args:
            mathematical_query: The original mathematical query or question
            domain: Optional mathematical domain classification
            
        Returns:
            List of search queries optimized for mathematical content
        """
        # Clean the query
        cleaned_query = self._clean_query(mathematical_query)
        
        # Detect domain if not provided
        if not domain:
            domain = classify_mathematical_domain(mathematical_query)
        
        # Extract key terms
        key_terms = self._extract_key_terms(cleaned_query)
        
        # Get domain-specific qualifiers
        domain_qualifiers = self._get_domain_qualifiers(domain)
        
        # Generate multiple query variants
        queries = []
        
        # Original query with domain qualifier (if available)
        if domain and domain in self.domain_qualifiers:
            qualifier = self.domain_qualifiers[domain][0]
            queries.append(f"{cleaned_query} {qualifier}")
        else:
            queries.append(cleaned_query)
        
        # Key terms with domain qualifier
        if key_terms:
            key_terms_str = " ".join(key_terms)
            if domain and domain in self.domain_qualifiers:
                qualifier = self.domain_qualifiers[domain][0]
                queries.append(f"{key_terms_str} {qualifier}")
            else:
                queries.append(key_terms_str)
        
        # Step-by-step solution query
        queries.append(f"how to {cleaned_query} step by step")
        
        # Definition query (if it seems like a concept question)
        if any(term in cleaned_query.lower() for term in ["what is", "define", "meaning of", "concept of"]):
            # Extract the concept being asked about
            concept = re.sub(r'(what is|define|meaning of|concept of)\s+', '', cleaned_query.lower())
            queries.append(f"definition of {concept} in mathematics")
        
        # Theorem query (if it seems like a theorem question)
        if any(term in cleaned_query.lower() for term in ["theorem", "proof", "lemma"]):
            queries.append(f"{cleaned_query} mathematical proof")
        
        # Academic search query for more advanced topics
        academic_query = f"{cleaned_query} mathematical paper"
        queries.append(academic_query)
        
        # Remove duplicates while preserving order
        unique_queries = []
        for query in queries:
            if query not in unique_queries:
                unique_queries.append(query)
        
        logger.info(f"Generated {len(unique_queries)} search queries for: {mathematical_query}")
        return unique_queries
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize a mathematical query.
        
        Args:
            query: Original query string
            
        Returns:
            Cleaned and normalized query
        """
        # Remove unnecessary words
        filler_words = [
            "can you", "please", "could you", "i want to", "i need to",
            "help me", "solve", "find", "calculate", "compute"
        ]
        
        cleaned = query.lower()
        for word in filler_words:
            cleaned = cleaned.replace(word, "")
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove trailing punctuation
        cleaned = re.sub(r'[.,?!]$', '', cleaned)
        
        return cleaned
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key mathematical terms from the query.
        
        Args:
            query: Cleaned query string
            
        Returns:
            List of key mathematical terms
        """
        # Convert to lowercase for matching
        query_lower = query.lower()
        
        # Extract matches for known mathematical keywords
        matches = []
        for keyword in self.math_keywords:
            if keyword in query_lower:
                matches.append(keyword)
        
        # Extract mathematical expressions (assuming they're in LaTeX format)
        latex_expressions = re.findall(r'\$(.*?)\$', query)
        if latex_expressions:
            matches.extend(latex_expressions)
        
        # Extract capitalized terms (likely proper nouns like "Fourier transform")
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        matches.extend(proper_nouns)
        
        # If we have too few matches, fall back to important words
        if len(matches) < 2:
            # Split the query into words
            words = query.split()
            
            # Filter out common stop words
            stop_words = ["the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "by"]
            important_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
            
            matches.extend(important_words)
        
        # Remove duplicates
        unique_matches = []
        for match in matches:
            if match not in unique_matches:
                unique_matches.append(match)
        
        return unique_matches
    
    def _get_domain_qualifiers(self, domain: Optional[str]) -> List[str]:
        """
        Get domain-specific search qualifiers.
        
        Args:
            domain: Mathematical domain
            
        Returns:
            List of domain-specific qualifiers
        """
        if not domain or domain not in self.domain_qualifiers:
            return []
        
        return self.domain_qualifiers[domain]
    
    def generate_specialized_queries(self, mathematical_query: str, domain: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Generate specialized queries for different search engines.
        
        Args:
            mathematical_query: The original mathematical query or question
            domain: Optional mathematical domain classification
            
        Returns:
            Dictionary mapping search engine types to optimized queries
        """
        # Generate base queries
        base_queries = self.generate_queries(mathematical_query, domain)
        
        specialized_queries = {
            "general": base_queries,
            "academic": [],
            "computational": []
        }
        
        # Generate academic queries (for arXiv, Google Scholar, etc.)
        for query in base_queries[:2]:  # Use only the top 2 base queries
            academic_query = f"{query} mathematical research"
            specialized_queries["academic"].append(academic_query)
        
        # Add mathematical notation if available
        latex_expressions = re.findall(r'\$(.*?)\$', mathematical_query)
        if latex_expressions:
            for expr in latex_expressions:
                # Try to parse the expression for Wolfram Alpha
                try:
                    parsed = parse_math_expression(expr)
                    if parsed["success"]:
                        wolfram_expr = str(parsed["expression"])
                        specialized_queries["computational"].append(wolfram_expr)
                except Exception as e:
                    logger.warning(f"Failed to parse LaTeX expression: {expr}, error: {str(e)}")
                    # Fall back to the original expression
                    specialized_queries["computational"].append(expr)
        else:
            # If no LaTeX, use the first base query for computational searches
            specialized_queries["computational"] = [base_queries[0]]
        
        return specialized_queries


class QueryOptimizer:
    """
    Optimizer for mathematical search queries.
    
    This class refines search queries based on initial results and feedback
    to improve the relevance of mathematical information retrieval.
    """
    
    def __init__(self, query_generator: MathSearchQueryGenerator):
        """
        Initialize the query optimizer.
        
        Args:
            query_generator: An instance of MathSearchQueryGenerator
        """
        self.query_generator = query_generator
    
    def refine_query(self, original_query: str, initial_results: Dict[str, Any]) -> List[str]:
        """
        Refine a query based on initial search results.
        
        Args:
            original_query: The original search query
            initial_results: Results from the initial search
            
        Returns:
            List of refined queries
        """
        # Check if we have successful results
        if not initial_results.get("success", False) or not initial_results.get("items", []):
            # If no results, generate more generic queries
            return self._generate_fallback_queries(original_query)
        
        # Extract common terms from result titles and snippets
        common_terms = self._extract_common_terms(initial_results)
        
        # Generate refined queries using common terms
        refined_queries = []
        
        # Add most common term to original query
        if common_terms:
            refined_queries.append(f"{original_query} {common_terms[0]}")
        
        # Add domain-specific refinements if detected
        domain = self._detect_domain_from_results(initial_results)
        if domain:
            domain_qualifiers = self.query_generator._get_domain_qualifiers(domain)
            if domain_qualifiers:
                refined_queries.append(f"{original_query} {domain_qualifiers[0]}")
        
        # If no refined queries were generated, fall back to original
        if not refined_queries:
            refined_queries.append(original_query)
        
        return refined_queries
    
    def _generate_fallback_queries(self, original_query: str) -> List[str]:
        """
        Generate fallback queries when initial search fails.
        
        Args:
            original_query: The original search query
            
        Returns:
            List of fallback queries
        """
        fallback_queries = []
        
        # Try more general mathematical terms
        fallback_queries.append(f"mathematics {original_query}")
        
        # Try educational resources
        fallback_queries.append(f"{original_query} tutorial")
        fallback_queries.append(f"{original_query} example problems")
        
        # Try removing potential complexity
        simplified_query = re.sub(r'\$(.*?)\$', '', original_query)  # Remove LaTeX
        simplified_query = re.sub(r'\s+', ' ', simplified_query).strip()  # Clean up whitespace
        if simplified_query and simplified_query != original_query:
            fallback_queries.append(simplified_query)
        
        return fallback_queries
    
    def _extract_common_terms(self, search_results: Dict[str, Any]) -> List[str]:
        """
        Extract common terms from search results.
        
        Args:
            search_results: Results from a search
            
        Returns:
            List of common terms ordered by frequency
        """
        # Extract text from titles and snippets
        all_text = ""
        for item in search_results.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            all_text += f" {title} {snippet}"
        
        # Convert to lowercase
        all_text = all_text.lower()
        
        # Remove common words and punctuation
        stop_words = [
            "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "by",
            "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
            "those", "it", "its", "mathematical", "mathematics", "math"
        ]
        
        # Replace punctuation with spaces
        for char in ".,;:!?()[]{}\"'":
            all_text = all_text.replace(char, " ")
        
        # Split into words and count frequencies
        words = all_text.split()
        word_counts = {}
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top words
        return [word for word, count in sorted_words[:5]]
    
    def _detect_domain_from_results(self, search_results: Dict[str, Any]) -> Optional[str]:
        """
        Detect mathematical domain from search results.
        
        Args:
            search_results: Results from a search
            
        Returns:
            Detected mathematical domain or None
        """
        # Extract text from titles and snippets
        all_text = ""
        for item in search_results.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            all_text += f" {title} {snippet}"
        
        # Check for domain-specific keywords
        domain_keywords = {
            "algebra": ["algebra", "equation", "polynomial", "factor", "root", "solve"],
            "calculus": ["calculus", "derivative", "integral", "differentiation", "integration", "limit"],
            "linear_algebra": ["matrix", "vector", "linear", "eigenvalue", "eigenvector", "determinant"],
            "statistics": ["statistics", "probability", "distribution", "random", "hypothesis", "test"],
            "geometry": ["geometry", "coordinate", "angle", "triangle", "circle", "polygon"]
        }
        
        # Count occurrences of domain keywords
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, all_text.lower())
                domain_scores[domain] += len(matches)
        
        # Find domain with highest score
        max_score = 0
        detected_domain = None
        
        for domain, score in domain_scores.items():
            if score > max_score:
                max_score = score
                detected_domain = domain
        
        # Only return domain if score is significant
        if max_score >= 3:
            return detected_domain
        
        return None
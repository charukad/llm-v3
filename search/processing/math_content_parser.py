"""
Mathematical content extraction from search results.

This module provides functionality to extract and process mathematical content 
from web pages and other sources retrieved by the Search Agent.
"""

import re
import requests
import logging
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class MathContentExtractor:
    """
    Extracts mathematical content from web pages and documents.
    
    This class provides methods to extract mathematical formulas, definitions,
    theorems, and explanations from various online sources.
    """
    
    def __init__(self):
        """Initialize the math content extractor."""
        # Regular expressions for identifying mathematical content
        self.latex_pattern = re.compile(r'(?:\\\[|\\\(|\$\$|\$)(.*?)(?:\\\]|\\\)|\$\$|\$)')
        self.equation_pattern = re.compile(r'<(?:div|span)[^>]*?(?:equation|math|formula)[^>]*?>(.*?)</(?:div|span)>', re.DOTALL)
        
        # Patterns for identifying theorems, definitions, etc.
        self.theorem_pattern = re.compile(r'(?:<(?:div|p|span)[^>]*?(?:theorem|lemma|corollary)[^>]*?>|(?:Theorem|THEOREM|Lemma|LEMMA|Corollary|COROLLARY)\s+[\d\.]+:)(.*?)(?:</(?:div|p|span)>|(?=<h))', re.DOTALL)
        self.definition_pattern = re.compile(r'(?:<(?:div|p|span)[^>]*?(?:definition)[^>]*?>|(?:Definition|DEFINITION)\s+[\d\.]+:)(.*?)(?:</(?:div|p|span)>|(?=<h))', re.DOTALL)
        self.proof_pattern = re.compile(r'(?:<(?:div|p|span)[^>]*?(?:proof)[^>]*?>|(?:Proof|PROOF):)(.*?)(?:</(?:div|p|span)>|(?=<h)|\\square|□)', re.DOTALL)
        self.example_pattern = re.compile(r'(?:<(?:div|p|span)[^>]*?(?:example)[^>]*?>|(?:Example|EXAMPLE)\s+[\d\.]+:)(.*?)(?:</(?:div|p|span)>|(?=<h))', re.DOTALL)
        
        # User agent for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract mathematical content from a web page.
        
        Args:
            url: URL of the web page
            
        Returns:
            Dictionary containing extracted mathematical content
        """
        try:
            logger.info(f"Extracting content from URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content
            content = {
                "latex_expressions": self._extract_latex(soup, response.text),
                "theorems": self._extract_theorems(soup, response.text),
                "definitions": self._extract_definitions(soup, response.text),
                "proofs": self._extract_proofs(soup, response.text),
                "examples": self._extract_examples(soup, response.text),
                "main_text": self._extract_main_content(soup),
                "title": self._extract_title(soup),
                "source": url
            }
            
            return {
                "success": True,
                "content": content
            }
            
        except Exception as e:
            logger.exception(f"Error extracting content from {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def extract_from_search_results(self, search_results: Dict[str, Any], max_urls: int = 3) -> Dict[str, Any]:
        """
        Extract content from multiple search results.
        
        Args:
            search_results: Search results containing URLs to process
            max_urls: Maximum number of URLs to process
            
        Returns:
            Dictionary containing aggregated mathematical content
        """
        if not search_results.get("success", False) or not search_results.get("items", []):
            return {
                "success": False,
                "error": "No search results to process",
                "aggregated_content": {}
            }
        
        # Extract URLs from search results
        urls = []
        for item in search_results.get("items", []):
            link = item.get("link", "")
            if link and link not in urls:
                urls.append(link)
                
                if len(urls) >= max_urls:
                    break
        
        # Process each URL
        content_results = []
        for url in urls:
            # Add a small delay to avoid overwhelming servers
            time.sleep(1)
            result = self.extract_from_url(url)
            
            if result.get("success", False):
                content_results.append(result.get("content", {}))
        
        # Aggregate content
        aggregated = self._aggregate_content(content_results)
        
        return {
            "success": True,
            "aggregated_content": aggregated,
            "individual_results": content_results,
            "query": search_results.get("query", "")
        }
    
    def _extract_latex(self, soup: BeautifulSoup, raw_html: str) -> List[str]:
        """
        Extract LaTeX expressions from HTML.
        
        Args:
            soup: BeautifulSoup object
            raw_html: Raw HTML text
            
        Returns:
            List of LaTeX expressions
        """
        expressions = []
        
        # Find expressions in MathJax/KaTeX elements
        math_elements = soup.find_all(["script", "span"], {"class": ["math", "MathJax", "katex"]})
        for element in math_elements:
            if element.string:
                expressions.append(element.string.strip())
        
        # Find expressions using regex patterns
        latex_matches = self.latex_pattern.findall(raw_html)
        expressions.extend(latex_matches)
        
        # Find expressions in equation divs/spans
        equation_matches = self.equation_pattern.findall(raw_html)
        expressions.extend(equation_matches)
        
        # Clean and deduplicate expressions
        cleaned_expressions = []
        for expr in expressions:
            cleaned = self._clean_latex(expr)
            if cleaned and cleaned not in cleaned_expressions:
                cleaned_expressions.append(cleaned)
        
        return cleaned_expressions
    
    def _clean_latex(self, latex: str) -> str:
        """
        Clean a LaTeX expression.
        
        Args:
            latex: Raw LaTeX expression
            
        Returns:
            Cleaned LaTeX expression
        """
        # Remove unnecessary whitespace
        cleaned = re.sub(r'\s+', ' ', latex).strip()
        
        # Remove certain HTML entities
        cleaned = cleaned.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        
        return cleaned
    
    def _extract_theorems(self, soup: BeautifulSoup, raw_html: str) -> List[Dict[str, str]]:
        """
        Extract theorems from HTML.
        
        Args:
            soup: BeautifulSoup object
            raw_html: Raw HTML text
            
        Returns:
            List of theorem dictionaries
        """
        theorems = []
        
        # Find theorem elements using class or id attributes
        theorem_elements = soup.find_all(["div", "p", "span"], class_=lambda c: c and any(x in c.lower() for x in ["theorem", "lemma", "corollary"]))
        theorem_elements.extend(soup.find_all(["div", "p", "span"], id=lambda i: i and any(x in i.lower() for x in ["theorem", "lemma", "corollary"])))
        
        for element in theorem_elements:
            title = element.find(["h2", "h3", "h4", "strong", "b"]).get_text() if element.find(["h2", "h3", "h4", "strong", "b"]) else ""
            content = element.get_text()
            if title:
                content = content.replace(title, "", 1).strip()
            
            theorems.append({
                "title": title,
                "content": content
            })
        
        # Find theorems using regex pattern
        theorem_matches = self.theorem_pattern.findall(raw_html)
        for match in theorem_matches:
            # Extract content and remove HTML tags
            content = BeautifulSoup(match, 'html.parser').get_text().strip()
            
            # Extract title if present (e.g., "Theorem 1.2")
            title_match = re.search(r'(?:Theorem|THEOREM|Lemma|LEMMA|Corollary|COROLLARY)\s+[\d\.]+', content)
            title = title_match.group(0) if title_match else ""
            
            if title:
                content = content.replace(title, "", 1).strip()
            
            theorems.append({
                "title": title,
                "content": content
            })
        
        # Deduplicate theorems
        unique_theorems = []
        for theorem in theorems:
            if not any(t["content"] == theorem["content"] for t in unique_theorems):
                unique_theorems.append(theorem)
        
        return unique_theorems
    
    def _extract_definitions(self, soup: BeautifulSoup, raw_html: str) -> List[Dict[str, str]]:
        """
        Extract definitions from HTML.
        
        Args:
            soup: BeautifulSoup object
            raw_html: Raw HTML text
            
        Returns:
            List of definition dictionaries
        """
        definitions = []
        
        # Find definition elements using class or id attributes
        definition_elements = soup.find_all(["div", "p", "span"], class_=lambda c: c and "definition" in c.lower())
        definition_elements.extend(soup.find_all(["div", "p", "span"], id=lambda i: i and "definition" in i.lower()))
        
        for element in definition_elements:
            title = element.find(["h2", "h3", "h4", "strong", "b"]).get_text() if element.find(["h2", "h3", "h4", "strong", "b"]) else ""
            content = element.get_text()
            if title:
                content = content.replace(title, "", 1).strip()
            
            definitions.append({
                "title": title,
                "content": content
            })
        
        # Find definitions using regex pattern
        definition_matches = self.definition_pattern.findall(raw_html)
        for match in definition_matches:
            # Extract content and remove HTML tags
            content = BeautifulSoup(match, 'html.parser').get_text().strip()
            
            # Extract title if present (e.g., "Definition 1.2")
            title_match = re.search(r'(?:Definition|DEFINITION)\s+[\d\.]+', content)
            title = title_match.group(0) if title_match else ""
            
            if title:
                content = content.replace(title, "", 1).strip()
            
            definitions.append({
                "title": title,
                "content": content
            })
        
        # Deduplicate definitions
        unique_definitions = []
        for definition in definitions:
            if not any(d["content"] == definition["content"] for d in unique_definitions):
                unique_definitions.append(definition)
        
        return unique_definitions
    
    def _extract_proofs(self, soup: BeautifulSoup, raw_html: str) -> List[str]:
        """
        Extract proofs from HTML.
        
        Args:
            soup: BeautifulSoup object
            raw_html: Raw HTML text
            
        Returns:
            List of proof strings
        """
        proofs = []
        
        # Find proof elements using class or id attributes
        proof_elements = soup.find_all(["div", "p", "span"], class_=lambda c: c and "proof" in c.lower())
        proof_elements.extend(soup.find_all(["div", "p", "span"], id=lambda i: i and "proof" in i.lower()))
        
        for element in proof_elements:
            proofs.append(element.get_text().strip())
        
        # Find proofs using regex pattern
        proof_matches = self.proof_pattern.findall(raw_html)
        for match in proof_matches:
            # Extract content and remove HTML tags
            content = BeautifulSoup(match, 'html.parser').get_text().strip()
            proofs.append(content)
        
        # Deduplicate proofs
        unique_proofs = []
        for proof in proofs:
            if proof not in unique_proofs:
                unique_proofs.append(proof)
        
        return unique_proofs
    
    def _extract_examples(self, soup: BeautifulSoup, raw_html: str) -> List[str]:
        """
        Extract examples from HTML.
        
        Args:
            soup: BeautifulSoup object
            raw_html: Raw HTML text
            
        Returns:
            List of example strings
        """
        examples = []
        
        # Find example elements using class or id attributes
        example_elements = soup.find_all(["div", "p", "span"], class_=lambda c: c and "example" in c.lower())
        example_elements.extend(soup.find_all(["div", "p", "span"], id=lambda i: i and "example" in i.lower()))
        
        for element in example_elements:
            examples.append(element.get_text().strip())
        
        # Find examples using regex pattern
        example_matches = self.example_pattern.findall(raw_html)
        for match in example_matches:
            # Extract content and remove HTML tags
            content = BeautifulSoup(match, 'html.parser').get_text().strip()
            examples.append(content)
        
        # Deduplicate examples
        unique_examples = []
        for example in examples:
            if example not in unique_examples:
                unique_examples.append(example)
        
        return unique_examples
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Main content text
        """
        # Try to find main content container
        main_content = None
        
        # Common content containers
        for selector in ["article", "main", ".content", "#content", ".main", "#main", ".post", "#post"]:
            if not main_content:
                main_content = soup.select_one(selector)
        
        # If no main content container found, use body
        if not main_content:
            main_content = soup.body
        
        # Remove unwanted elements
        if main_content:
            for element in main_content.find_all(["nav", "header", "footer", "aside", "script", "style", "noscript", "iframe"]):
                element.decompose()
            
            return main_content.get_text().strip()
        
        return ""
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract title from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Page title
        """
        # Try to find title
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.string.strip()
        
        # Try to find h1
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return ""
    
    def _aggregate_content(self, content_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate content from multiple sources.
        
        Args:
            content_results: List of content dictionaries from different sources
            
        Returns:
            Aggregated content dictionary
        """
        aggregated = {
            "latex_expressions": [],
            "theorems": [],
            "definitions": [],
            "proofs": [],
            "examples": [],
            "sources": []
        }
        
        for content in content_results:
            # Add unique LaTeX expressions
            for expr in content.get("latex_expressions", []):
                if expr not in aggregated["latex_expressions"]:
                    aggregated["latex_expressions"].append(expr)
            
            # Add unique theorems
            for theorem in content.get("theorems", []):
                if not any(t["content"] == theorem["content"] for t in aggregated["theorems"]):
                    aggregated["theorems"].append(theorem)
            
            # Add unique definitions
            for definition in content.get("definitions", []):
                if not any(d["content"] == definition["content"] for d in aggregated["definitions"]):
                    aggregated["definitions"].append(definition)
            
            # Add unique proofs
            for proof in content.get("proofs", []):
                if proof not in aggregated["proofs"]:
                    aggregated["proofs"].append(proof)
            
            # Add unique examples
            for example in content.get("examples", []):
                if example not in aggregated["examples"]:
                    aggregated["examples"].append(example)
            
            # Add source
            if content.get("source") and content.get("title"):
                aggregated["sources"].append({
                    "url": content["source"],
                    "title": content["title"]
                })
        
        return aggregated


class MathCredibilityEvaluator:
    """
    Evaluates the credibility of mathematical content sources.
    
    This class assesses the reliability and authority of mathematical information
    sources to ensure that the Search Agent provides trustworthy information.
    """
    
    def __init__(self):
        """Initialize the credibility evaluator."""
        # High credibility domains for mathematical content
        self.high_credibility_domains = [
            "arxiv.org",
            "mathworld.wolfram.com",
            "encyclopediaofmath.org",
            "ams.org",
            "math.stackexchange.com",
            "mathoverflow.net",
            "projecteuclid.org",
            "maa.org",
            "siam.org",
            "aps.org",
            "nist.gov",
            "mit.edu",
            "stanford.edu",
            "harvard.edu",
            "princeton.edu",
            "berkeley.edu",
            "cam.ac.uk",
            "ox.ac.uk",
            "waterloo.ca",
            "khanacademy.org",
            "brilliant.org",
            "paulgraham.com",
            "wikipedia.org"  # While not perfect, often reliable for mathematical topics
        ]
        
        # Medium credibility domains
        self.medium_credibility_domains = [
            "mathsisfun.com",
            "purplemath.com",
            "mathcentre.ac.uk",
            "sosmath.com",
            "mathplanet.com",
            "mathway.com",
            "symbolab.com",
            "wolframalpha.com",
            "chegg.com",
            "coursera.org",
            "edx.org",
            "udacity.com",
            "openstudy.com",
            "desmos.com"
        ]
    
    def evaluate_source(self, url: str) -> Dict[str, Any]:
        """
        Evaluate the credibility of a source.
        
        Args:
            url: URL of the source
            
        Returns:
            Dictionary containing credibility evaluation
        """
        # Extract domain from URL
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if not domain_match:
            return {
                "credibility": "unknown",
                "score": 0.0,
                "reason": "Unable to extract domain from URL"
            }
        
        domain = domain_match.group(1)
        
        # Check if domain is in high credibility list
        for high_domain in self.high_credibility_domains:
            if high_domain in domain:
                return {
                    "credibility": "high",
                    "score": 0.9,
                    "reason": f"Domain {domain} is a highly credible source for mathematical content"
                }
        
        # Check if domain is in medium credibility list
        for medium_domain in self.medium_credibility_domains:
            if medium_domain in domain:
                return {
                    "credibility": "medium",
                    "score": 0.6,
                    "reason": f"Domain {domain} is a moderately credible source for mathematical content"
                }
        
        # Check for educational or academic domains
        if domain.endswith(".edu") or domain.endswith(".ac.uk") or domain.endswith(".ac.jp"):
            return {
                "credibility": "high",
                "score": 0.85,
                "reason": f"Domain {domain} is an educational institution"
            }
        
        # Check for government domains
        if domain.endswith(".gov"):
            return {
                "credibility": "high",
                "score": 0.8,
                "reason": f"Domain {domain} is a government institution"
            }
        
        # Default to low credibility for unknown sources
        return {
            "credibility": "low",
            "score": 0.3,
            "reason": f"Domain {domain} is not a recognized source for mathematical content"
        }
    
    def evaluate_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the credibility of content based on various factors.
        
        Args:
            content: Content dictionary from MathContentExtractor
            
        Returns:
            Dictionary containing credibility evaluation
        """
        # Start with source credibility
        source_evaluation = self.evaluate_source(content.get("source", ""))
        initial_score = source_evaluation.get("score", 0.0)
        
        # Adjust score based on content factors
        content_score = initial_score
        
        # Presence of LaTeX expressions (indicates mathematical rigor)
        if len(content.get("latex_expressions", [])) > 0:
            content_score += 0.1
        
        # Presence of theorems and proofs (indicates mathematical depth)
        if len(content.get("theorems", [])) > 0 and len(content.get("proofs", [])) > 0:
            content_score += 0.15
        
        # Presence of definitions (indicates clarity)
        if len(content.get("definitions", [])) > 0:
            content_score += 0.05
        
        # Presence of examples (indicates practical understanding)
        if len(content.get("examples", [])) > 0:
            content_score += 0.05
        
        # Cap the score at 1.0
        content_score = min(content_score, 1.0)
        
        # Determine overall credibility level
        credibility_level = "low"
        if content_score >= 0.7:
            credibility_level = "high"
        elif content_score >= 0.4:
            credibility_level = "medium"
        
        return {
            "credibility": credibility_level,
            "score": content_score,
            "source_evaluation": source_evaluation,
            "factors": {
                "has_latex": len(content.get("latex_expressions", [])) > 0,
                "has_theorems": len(content.get("theorems", [])) > 0,
                "has_proofs": len(content.get("proofs", [])) > 0,
                "has_definitions": len(content.get("definitions", [])) > 0,
                "has_examples": len(content.get("examples", [])) > 0
            }
        }


class RelevanceScoringEngine:
    """
    Scores search results for relevance to the original mathematical query.
    
    This class provides methods to rank and filter search results based on
    their relevance to the original mathematical question.
    """
    
    def __init__(self):
        """Initialize the relevance scoring engine."""
        # Common mathematical terms for relevance assessment
        self.math_terms = [
            "theorem", "proof", "lemma", "corollary", "definition", "equation",
            "formula", "integral", "derivative", "matrix", "vector", "function",
            "algebra", "calculus", "topology", "geometry", "statistics", "probability",
            "linear", "nonlinear", "differential", "set", "group", "ring", "field",
            "limit", "convergence", "series", "sequence", "metric", "space", "transformation"
        ]
    
    def score_content(self, content: Dict[str, Any], query: str) -> float:
        """
        Score content relevance to a query.
        
        Args:
            content: Content dictionary from MathContentExtractor
            query: Original query string
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        query_terms = self._extract_query_terms(query)
        query_term_count = len(query_terms)
        
        if query_term_count == 0:
            return 0.0
        
        # Prepare content text
        content_text = " ".join([
            content.get("title", ""),
            content.get("main_text", ""),
            " ".join([t.get("content", "") for t in content.get("theorems", [])]),
            " ".join([d.get("content", "") for d in content.get("definitions", [])]),
            " ".join(content.get("proofs", [])),
            " ".join(content.get("examples", []))
        ]).lower()
        
        # Count matching terms
        matches = 0
        for term in query_terms:
            if term.lower() in content_text:
                matches += 1
        
        # Calculate basic relevance score
        term_match_score = matches / query_term_count
        
        # Adjust score based on presence of mathematical content
        math_content_score = 0.0
        
        # LaTeX expressions
        if len(content.get("latex_expressions", [])) > 0:
            math_content_score += 0.2
        
        # Theorems and proofs
        if len(content.get("theorems", [])) > 0:
            math_content_score += 0.2
        
        if len(content.get("proofs", [])) > 0:
            math_content_score += 0.2
        
        # Definitions and examples
        if len(content.get("definitions", [])) > 0:
            math_content_score += 0.2
        
        if len(content.get("examples", [])) > 0:
            math_content_score += 0.2
        
        # Cap math content score at 1.0
        math_content_score = min(math_content_score, 1.0)
        
        # Combined score (weight term matches and math content equally)
        combined_score = 0.6 * term_match_score + 0.4 * math_content_score
        
        return combined_score
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """
        Extract significant terms from a query.
        
        Args:
            query: Query string
            
        Returns:
            List of significant terms
        """
        # Remove common stop words
        stop_words = [
            "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "by",
            "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
            "those", "it", "its", "how", "what", "when", "where", "who", "which", "why",
            "can", "could", "would", "should", "will", "shall", "may", "might", "must"
        ]
        
        # Split into words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add mathematical terms that appear in the query
        for term in self.math_terms:
            if term.lower() in query.lower() and term not in filtered_words:
                filtered_words.append(term)
        
        return filtered_words
    
    def rank_search_results(self, search_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Rank search results by relevance.
        
        Args:
            search_results: Search results with extracted content
            query: Original query string
            
        Returns:
            Results dictionary with added relevance scores
        """
        # Check if results are available
        if not search_results.get("success", False) or not search_results.get("individual_results", []):
            return search_results
        
        # Score individual results
        scored_results = []
        for result in search_results.get("individual_results", []):
            score = self.score_content(result, query)
            scored_results.append({
                "content": result,
                "relevance_score": score
            })
        
        # Sort by relevance score (descending)
        sorted_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
        
        # Update search results with ranked individual results
        search_results["scored_results"] = sorted_results
        
        # Regenerate aggregated content using top results only
        top_results = [r["content"] for r in sorted_results if r["relevance_score"] > 0.3]
        if top_results:
            extractor = MathContentExtractor()
            search_results["aggregated_content"] = extractor._aggregate_content(top_results)
        
        return search_results
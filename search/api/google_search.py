"""
Google Search API integration for the Search Agent.

This module provides functionality to interact with the Google Custom Search API
to retrieve mathematical information from the web.
"""

import requests
import logging
import os
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class GoogleSearchAPI:
    """
    Client for the Google Custom Search JSON API.
    
    This class handles authentication, request formatting, and parsing responses
    from the Google Custom Search API.
    """
    
    def __init__(self, api_key: Optional[str] = None, cx: Optional[str] = None):
        """
        Initialize the Google Search API client.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY environment variable)
            cx: Google Custom Search Engine ID (defaults to GOOGLE_CX environment variable)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.cx = cx or os.environ.get("GOOGLE_CX")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key:
            logger.warning("Google API key not provided. Search functionality will be limited.")
        
        if not self.cx:
            logger.warning("Google Custom Search Engine ID not provided. Search functionality will be limited.")
    
    def search(self, query: str, num_results: int = 5, page: int = 1) -> Dict[str, Any]:
        """
        Perform a Google search with the given query.
        
        Args:
            query: The search query string
            num_results: Number of results to return (max 10 per request)
            page: Result page number (for pagination)
            
        Returns:
            Dictionary containing search results or error information
        """
        if not self.api_key or not self.cx:
            return {
                "success": False,
                "error": "Missing API key or Custom Search Engine ID",
                "items": []
            }
        
        # Ensure num_results is within Google's limits (1-10)
        num_results = min(max(1, num_results), 10)
        
        # Calculate start index for pagination (1-based indexing)
        start_index = (page - 1) * num_results + 1
        
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": num_results,
            "start": start_index,
            # Prioritize reliable academic and educational sources for mathematical content
            "sort": "date-scy",  # Sort by date with scientific information first
            "safe": "active"
        }
        
        try:
            logger.info(f"Performing Google search for query: {query}")
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Received {len(results.get('items', []))} search results")
                return {
                    "success": True,
                    "items": results.get("items", []),
                    "search_information": results.get("searchInformation", {}),
                    "query": query
                }
            elif response.status_code == 429:  # Rate limit exceeded
                logger.warning("Google API rate limit exceeded")
                time.sleep(2)  # Add a delay before the next request
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "items": []
                }
            else:
                logger.error(f"Google search failed with status code: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Search failed with status code: {response.status_code}",
                    "items": []
                }
                
        except Exception as e:
            logger.exception(f"Error performing Google search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "items": []
            }
    
    def batch_search(self, queries: List[str], num_results: int = 3) -> Dict[str, Any]:
        """
        Perform batch searches for multiple queries.
        
        Args:
            queries: List of search query strings
            num_results: Number of results per query
            
        Returns:
            Dictionary mapping queries to their search results
        """
        results = {}
        
        for query in queries:
            # Add a small delay between requests to avoid rate limiting
            time.sleep(1)
            results[query] = self.search(query, num_results=num_results)
        
        return results


class ArxivSearchAPI:
    """
    Client for the arXiv API to retrieve academic papers related to mathematics.
    
    This class provides functionality to search for mathematical research papers
    on arXiv, which is especially useful for advanced or recent mathematical topics.
    """
    
    def __init__(self):
        """Initialize the arXiv Search API client."""
        self.base_url = "http://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 5, start: int = 0) -> Dict[str, Any]:
        """
        Search arXiv for mathematical papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            start: Start index for results (for pagination)
            
        Returns:
            Dictionary containing search results or error information
        """
        # Add mathematics category constraint to focus on math papers
        # cat:math.* restricts to mathematics categories
        search_query = f"all:{query} AND (cat:math.* OR cat:cs.LG OR cat:stat.ML)"
        
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            logger.info(f"Searching arXiv for: {query}")
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                # arXiv returns XML, but for simplicity, we'll process it as text
                # In a production system, use a proper XML parser like ElementTree
                xml_response = response.text
                
                # Simple extraction of key information (this should be replaced with proper XML parsing)
                papers = []
                entry_blocks = xml_response.split("<entry>")[1:]
                
                for block in entry_blocks:
                    title = self._extract_tag_content(block, "title")
                    summary = self._extract_tag_content(block, "summary")
                    published = self._extract_tag_content(block, "published")
                    link = self._extract_arxiv_link(block)
                    
                    papers.append({
                        "title": title,
                        "summary": summary,
                        "published": published,
                        "link": link
                    })
                
                return {
                    "success": True,
                    "papers": papers,
                    "query": query
                }
            else:
                logger.error(f"arXiv search failed with status code: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Search failed with status code: {response.status_code}",
                    "papers": []
                }
                
        except Exception as e:
            logger.exception(f"Error performing arXiv search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "papers": []
            }
    
    def _extract_tag_content(self, text: str, tag: str) -> str:
        """
        Extract content between XML tags.
        
        Args:
            text: XML text
            tag: Tag name to extract
            
        Returns:
            Content between tags, or empty string if not found
        """
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_pos = text.find(start_tag)
        if start_pos == -1:
            return ""
        
        start_pos += len(start_tag)
        end_pos = text.find(end_tag, start_pos)
        
        if end_pos == -1:
            return ""
        
        return text[start_pos:end_pos].strip()
    
    def _extract_arxiv_link(self, text: str) -> str:
        """
        Extract arXiv paper link from entry XML.
        
        Args:
            text: Entry XML text
            
        Returns:
            arXiv paper link or empty string if not found
        """
        link_tag = '<link title="pdf" href="'
        start_pos = text.find(link_tag)
        
        if start_pos == -1:
            return ""
        
        start_pos += len(link_tag)
        end_pos = text.find('"', start_pos)
        
        if end_pos == -1:
            return ""
        
        return text[start_pos:end_pos].strip()


class WolframAlphaAPI:
    """
    Client for the Wolfram Alpha API to retrieve computational results.
    
    This class provides functionality to verify mathematical computations and
    retrieve information about mathematical concepts from Wolfram Alpha.
    """
    
    def __init__(self, app_id: Optional[str] = None):
        """
        Initialize the Wolfram Alpha API client.
        
        Args:
            app_id: Wolfram Alpha Application ID (defaults to WOLFRAM_APP_ID environment variable)
        """
        self.app_id = app_id or os.environ.get("WOLFRAM_APP_ID")
        self.base_url = "https://api.wolframalpha.com/v2/query"
        
        if not self.app_id:
            logger.warning("Wolfram Alpha App ID not provided. Computation verification will be limited.")
    
    def query(self, query: str, format: str = "plaintext") -> Dict[str, Any]:
        """
        Query Wolfram Alpha with a mathematical expression or question.
        
        Args:
            query: Mathematical expression or question
            format: Response format (plaintext, image, etc.)
            
        Returns:
            Dictionary containing query results or error information
        """
        if not self.app_id:
            return {
                "success": False,
                "error": "Missing Wolfram Alpha App ID",
                "pods": []
            }
        
        params = {
            "input": query,
            "appid": self.app_id,
            "format": format,
            "output": "json"
        }
        
        try:
            logger.info(f"Querying Wolfram Alpha for: {query}")
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract pods (sections of the Wolfram Alpha response)
                pods = []
                if "queryresult" in result and "pods" in result["queryresult"]:
                    for pod in result["queryresult"]["pods"]:
                        pod_data = {
                            "title": pod.get("title", ""),
                            "subpods": []
                        }
                        
                        for subpod in pod.get("subpods", []):
                            pod_data["subpods"].append({
                                "plaintext": subpod.get("plaintext", ""),
                                "img": subpod.get("img", {}).get("src", "") if "img" in subpod else ""
                            })
                        
                        pods.append(pod_data)
                
                return {
                    "success": True,
                    "pods": pods,
                    "query": query
                }
            else:
                logger.error(f"Wolfram Alpha query failed with status code: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Query failed with status code: {response.status_code}",
                    "pods": []
                }
                
        except Exception as e:
            logger.exception(f"Error querying Wolfram Alpha: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pods": []
            }
    
    def verify_computation(self, expression: str) -> Dict[str, Any]:
        """
        Verify a mathematical computation using Wolfram Alpha.
        
        Args:
            expression: Mathematical expression to verify
            
        Returns:
            Dictionary containing verification results
        """
        result = self.query(expression)
        
        if not result["success"]:
            return result
        
        # Look for the "Result" pod
        result_pod = next((pod for pod in result["pods"] if pod["title"] == "Result"), None)
        
        if result_pod and result_pod["subpods"]:
            return {
                "success": True,
                "verified": True,
                "result": result_pod["subpods"][0].get("plaintext", ""),
                "expression": expression
            }
        
        return {
            "success": True,
            "verified": False,
            "message": "No definitive result found",
            "expression": expression
        }
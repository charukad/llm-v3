"""
Tests for the Search Agent component.

This module provides test cases for the Search Agent functionality.
"""

import unittest
import logging
from unittest.mock import MagicMock, patch

from search.agent.search_agent import SearchAgent
from search.query.query_generator import MathSearchQueryGenerator
from search.processing.math_content_parser import MathContentExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSearchAgent(unittest.TestCase):
    """Test cases for the Search Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create SearchAgent with mock configuration
        self.config = {
            "google_api_key": "test_api_key",
            "google_cx": "test_cx",
            "wolfram_app_id": "test_app_id"
        }
        self.search_agent = SearchAgent(config=self.config)
        
        # Mock external API calls
        self.search_agent.google_search.search = MagicMock()
        self.search_agent.arxiv_search.search = MagicMock()
        self.search_agent.wolfram_alpha.query = MagicMock()
    
    @patch('search.api.google_search.GoogleSearchAPI.search')
    def test_search_basic(self, mock_search):
        """Test basic search functionality."""
        # Mock search results
        mock_search.return_value = {
            "success": True,
            "items": [
                {
                    "title": "Introduction to Calculus",
                    "link": "https://example.com/calculus",
                    "snippet": "Calculus is the mathematical study of continuous change."
                }
            ],
            "query": "what is calculus"
        }
        
        # Perform search
        result = self.search_agent.search("what is calculus", domain="calculus")
        
        # Check that search was called
        mock_search.assert_called()
        
        # Check result structure
        self.assertTrue(result.get("success", False))
        self.assertIn("formatted_content", result)
    
    def test_query_generation(self):
        """Test mathematical query generation."""
        query_generator = MathSearchQueryGenerator()
        
        # Generate queries for a calculus question
        queries = query_generator.generate_queries(
            "Find the derivative of x^2 sin(x)",
            domain="calculus"
        )
        
        # Check that queries were generated
        self.assertGreater(len(queries), 0)
        
        # Check that at least one query contains "derivative"
        self.assertTrue(any("derivative" in q.lower() for q in queries))
    
    @patch('requests.get')
    def test_content_extraction(self, mock_get):
        """Test content extraction from search results."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Calculus Tutorial</title></head>
            <body>
                <h1>Introduction to Calculus</h1>
                <div class="definition">
                    <h2>Definition of Derivative</h2>
                    <p>The derivative of a function represents its rate of change.</p>
                </div>
                <div class="theorem">
                    <h2>Fundamental Theorem of Calculus</h2>
                    <p>The Fundamental Theorem of Calculus establishes the connection between differentiation and integration.</p>
                </div>
                <div class="math">
                    $f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        # Create content extractor
        extractor = MathContentExtractor()
        
        # Extract content
        result = extractor.extract_from_url("https://example.com/calculus")
        
        # Check that content was extracted
        self.assertTrue(result.get("success", False))
        self.assertIn("content", result)
        
        # Check that specific content types were extracted
        content = result.get("content", {})
        self.assertIn("title", content)
        self.assertEqual(content["title"], "Calculus Tutorial")
        
        # Check for definitions and theorems
        self.assertTrue(len(content.get("definitions", [])) > 0 or len(content.get("theorems", [])) > 0)
    
    def test_message_handling(self):
        """Test handling of MCP messages."""
        # Create a search request message
        message = {
            "header": {
                "message_id": "test_msg_001",
                "sender": "orchestration_manager",
                "recipient": "search_agent",
                "timestamp": 1234567890,
                "message_type": "search_request"
            },
            "body": {
                "query": "What is the derivative of x^2?",
                "domain": "calculus",
                "max_results": 3
            }
        }
        
        # Mock search method
        self.search_agent.search = MagicMock()
        self.search_agent.search.return_value = {
            "success": True,
            "message": "Found information",
            "formatted_content": {
                "context": "The derivative of x^2 is 2x.",
                "sources": [{"url": "https://example.com", "title": "Calculus Reference"}]
            }
        }
        
        # Handle message
        response = self.search_agent.handle_message(message)
        
        # Check response structure
        self.assertIn("header", response)
        self.assertIn("body", response)
        
        # Check that search was called with correct parameters
        self.search_agent.search.assert_called_with(
            "What is the derivative of x^2?",
            domain="calculus",
            max_results=3
        )
        
        # Check response header
        header = response.get("header", {})
        self.assertEqual(header.get("message_type"), "search_response")
        self.assertEqual(header.get("recipient"), "orchestration_manager")

if __name__ == '__main__':
    unittest.main()
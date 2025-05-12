"""
Tests for the LLM Connector component.

This module provides test cases for the integration of search results with the LLM.
"""

import unittest
from unittest.mock import MagicMock, patch

from search.agent.search_agent import SearchAgent
from search.integration.llm_connector import LLMConnector

class TestLLMConnector(unittest.TestCase):
    """Test cases for the LLM Connector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock SearchAgent
        self.search_agent = MagicMock(spec=SearchAgent)
        
        # Create LLMConnector with mock agent
        self.llm_connector = LLMConnector(search_agent=self.search_agent)
    
    def test_should_perform_search(self):
        """Test search decision logic."""
        # Mathematical questions should trigger search
        self.assertTrue(
            self.llm_connector._should_perform_search("What is the derivative of x^2?")
        )
        self.assertTrue(
            self.llm_connector._should_perform_search("Prove the Pythagorean theorem")
        )
        
        # Specialized queries should trigger search
        self.assertTrue(
            self.llm_connector._should_perform_search("Who discovered the Mandelbrot set?")
        )
        
        # Non-mathematical questions should not trigger search
        self.assertFalse(
            self.llm_connector._should_perform_search("Write a poem about math")
        )
    
    def test_enhance_prompt_with_search(self):
        """Test prompt enhancement with search results."""
        # Mock search results
        mock_search_results = {
            "success": True,
            "formatted_content": {
                "context": "The derivative of x^2 is 2x.",
                "sources": [{"url": "https://example.com", "title": "Calculus Reference"}]
            }
        }
        
        # Configure mock to return results
        self.search_agent.search.return_value = mock_search_results
        
        # Test prompt enhancement
        result = self.llm_connector.enhance_prompt("What is the derivative of x^2?", "calculus")
        
        # Verify search was called
        self.search_agent.search.assert_called_once()
        
        # Check result
        self.assertTrue(result["context_added"])
        self.assertIn("derivative of x^2 is 2x", result["enhanced_prompt"])
    
    def test_enhance_response_with_citations(self):
        """Test response enhancement with citations."""
        # Create sample search results
        search_results = {
            "success": True,
            "formatted_content": {
                "sources": [
                    {"url": "https://example.com", "title": "Math Reference"}
                ]
            }
        }
        
        # Original LLM output
        llm_output = "The derivative of x^2 is 2x."
        
        # Enhance response
        enhanced = self.llm_connector.enhance_response(
            llm_output, search_results, add_citations=True
        )
        
        # Check that citations were added
        self.assertIn("References:", enhanced)
        self.assertIn("Math Reference", enhanced)
    
    def test_build_enhanced_prompt(self):
        """Test enhanced prompt building."""
        original_prompt = "What is the derivative of x^2?"
        context_text = "The derivative of x^2 is 2x."
        
        enhanced = self.llm_connector._build_enhanced_prompt(original_prompt, context_text)
        
        # Check structure of enhanced prompt
        self.assertIn("ADDITIONAL INFORMATION:", enhanced)
        self.assertIn("USER QUERY:", enhanced)
        self.assertIn(original_prompt, enhanced)
        self.assertIn(context_text, enhanced)

if __name__ == '__main__':
    unittest.main()
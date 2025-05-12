#!/usr/bin/env python
"""
Test script for the Search Agent component.

This script demonstrates the capabilities of the Search Agent by performing
searches for various mathematical topics and displaying the results.
"""

import os
import sys
import logging
import argparse
from pprint import pprint

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search.agent.search_agent import SearchAgent
from search.integration.context_integrator import SearchContextIntegrator
from search.integration.citation_generator import CitationGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test the Search Agent functionality")
    
    parser.add_argument(
        "--query", 
        type=str, 
        help="Mathematical query to search for"
    )
    
    parser.add_argument(
        "--domain", 
        type=str, 
        choices=["algebra", "calculus", "linear_algebra", "statistics", "geometry", "number_theory", "discrete_math"],
        help="Mathematical domain for context"
    )
    
    parser.add_argument(
        "--citation-style", 
        type=str, 
        default="standard",
        choices=["standard", "apa", "mla", "chicago", "harvard", "ieee", "minimal"],
        help="Citation style for results"
    )
    
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Show detailed search results"
    )
    
    return parser.parse_args()

def setup_search_agent():
    """Set up and configure the Search Agent."""
    # Get API keys from environment
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_cx = os.environ.get("GOOGLE_CX")
    wolfram_app_id = os.environ.get("WOLFRAM_APP_ID")
    
    # Check for required API keys
    if not google_api_key or not google_cx:
        logger.warning("Missing Google API key or CX. Search functionality will be limited.")
    
    # Create configuration
    config = {
        "google_api_key": google_api_key,
        "google_cx": google_cx,
        "wolfram_app_id": wolfram_app_id
    }
    
    # Create and return SearchAgent
    return SearchAgent(config=config)

def display_results(search_results, citation_style="standard", detailed=False):
    """
    Display search results in a formatted way.
    
    Args:
        search_results: Search results from SearchAgent
        citation_style: Citation style for formatting
        detailed: Whether to show detailed results
    """
    if not search_results.get("success", False):
        logger.error(f"Search failed: {search_results.get('message', 'Unknown error')}")
        return
    
    formatted_content = search_results.get("formatted_content", {})
    
    # Display context summary
    print("\n" + "="*80)
    print("SEARCH RESULTS SUMMARY")
    print("="*80)
    
    if "context" in formatted_content:
        print(formatted_content["context"])
    
    # Display citations
    citation_generator = CitationGenerator()
    citations = citation_generator.generate_citations(search_results, citation_style)
    
    if citations:
        print("\nCITATIONS:")
        for i, citation in enumerate(citations, 1):
            print(f"{i}. {citation}")
    
    # Display detailed results if requested
    if detailed:
        print("\n" + "="*80)
        print("DETAILED SEARCH RESULTS")
        print("="*80)
        
        # Display LaTeX expressions
        if "latex_expressions" in formatted_content and formatted_content["latex_expressions"]:
            print("\nMATHEMATICAL EXPRESSIONS:")
            for expr in formatted_content["latex_expressions"]:
                print(f"${expr}$")
        
        # Display theorems
        if "theorems" in formatted_content and formatted_content["theorems"]:
            print("\nTHEOREMS:")
            for theorem in formatted_content["theorems"]:
                title = theorem.get("title", "")
                content = theorem.get("content", "")
                
                if title:
                    print(f"- {title}: {content}")
                else:
                    print(f"- {content}")
        
        # Display definitions
        if "definitions" in formatted_content and formatted_content["definitions"]:
            print("\nDEFINITIONS:")
            for definition in formatted_content["definitions"]:
                content = definition.get("content", "")
                print(f"- {content}")
        
        # Display examples
        if "examples" in formatted_content and formatted_content["examples"]:
            print("\nEXAMPLES:")
            for example in formatted_content["examples"]:
                print(f"- {example}")

def run_interactive_test():
    """Run an interactive test of the Search Agent."""
    search_agent = setup_search_agent()
    context_integrator = SearchContextIntegrator()
    
    print("="*80)
    print("SEARCH AGENT INTERACTIVE TEST")
    print("="*80)
    print("Type 'exit' to quit")
    print()
    
    while True:
        # Get query from user
        query = input("Enter a mathematical query: ")
        
        if query.lower() == 'exit':
            break
        
        # Get optional domain
        domain = input("Enter mathematical domain (optional, press Enter to skip): ")
        domain = domain if domain else None
        
        print("\nSearching...")
        
        # Perform search
        search_results = search_agent.search(query, domain)
        
        # Display results
        display_results(search_results, detailed=True)
        
        print("\n" + "-"*80)

def main():
    """Main entry point."""
    args = parse_args()
    search_agent = setup_search_agent()
    
    if args.query:
        # Use provided query
        logger.info(f"Searching for: {args.query}")
        
        search_results = search_agent.search(args.query, args.domain)
        display_results(search_results, args.citation_style, args.detailed)
    else:
        # Run interactive test
        run_interactive_test()

if __name__ == "__main__":
    main()
"""
Integration of search results with the LLM context.

This module provides functionality to integrate search results into the
context window of the Core LLM Agent for enhanced responses.
"""

import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SearchContextIntegrator:
    """
    Integrates search results into the LLM context.
    
    This class formats search results for inclusion in the LLM's context window,
    enhancing its responses with external information.
    """
    
    def __init__(self):
        """Initialize the search context integrator."""
        pass
    
    def integrate_search_results(self, search_results: Dict[str, Any], original_query: str) -> str:
        """
        Integrate search results into a context string for the LLM.
        
        Args:
            search_results: Formatted search results
            original_query: Original query string
            
        Returns:
            Formatted context string
        """
        if not search_results.get("success", False) or not search_results.get("formatted_content"):
            return ""
        
        formatted_content = search_results.get("formatted_content", {})
        
        # Use the pre-generated context if available
        if "context" in formatted_content and formatted_content["context"]:
            context = formatted_content["context"]
        else:
            # Build context from scratch
            context = f"Information from reliable sources about: {original_query}\n\n"
            
            # Add definitions if available
            if formatted_content.get("definitions"):
                context += "DEFINITIONS:\n"
                for definition in formatted_content["definitions"]:
                    definition_text = definition.get("content", "")
                    if definition_text:
                        context += f"- {definition_text}\n"
                context += "\n"
            
            # Add theorems if available
            if formatted_content.get("theorems"):
                context += "THEOREMS:\n"
                for theorem in formatted_content["theorems"]:
                    theorem_text = theorem.get("content", "")
                    theorem_title = theorem.get("title", "")
                    
                    if theorem_title and theorem_text:
                        context += f"- {theorem_title}: {theorem_text}\n"
                    elif theorem_text:
                        context += f"- {theorem_text}\n"
                context += "\n"
            
            # Add examples if available
            if formatted_content.get("examples"):
                context += "EXAMPLES:\n"
                for example in formatted_content["examples"]:
                    if example:
                        context += f"- {example}\n"
                context += "\n"
        
        # Add LaTeX expressions if available (for reference)
        if formatted_content.get("latex_expressions"):
            context += "\nRELEVANT MATHEMATICAL EXPRESSIONS:\n"
            for latex in formatted_content["latex_expressions"][:3]:
                if latex:
                    # Clean LaTeX for presentation
                    cleaned_latex = latex.replace("\n", " ").strip()
                    context += f"${cleaned_latex}$\n"
            context += "\n"
        
        # Add citation information
        if formatted_content.get("sources"):
            context += "\nSOURCES:\n"
            for source in formatted_content["sources"]:
                url = source.get("url", "")
                title = source.get("title", "Unknown Source")
                
                if url and title:
                    context += f"- {title}: {url}\n"
                elif url:
                    context += f"- {url}\n"
            context += "\n"
        
        # Add credibility note if available
        if "credibility" in formatted_content:
            credibility_score = formatted_content["credibility"]
            
            credibility_level = "low"
            if credibility_score >= 0.7:
                credibility_level = "high"
            elif credibility_score >= 0.4:
                credibility_level = "moderate"
            
            context += f"Note: This information has {credibility_level} credibility based on source evaluation.\n"
        
        return context
    
    def generate_citations(self, search_results: Dict[str, Any], citation_style: str = "standard") -> List[str]:
        """
        Generate citations for search results.
        
        Args:
            search_results: Formatted search results
            citation_style: Citation style (standard, academic, or minimal)
            
        Returns:
            List of citation strings
        """
        if not search_results.get("success", False) or not search_results.get("formatted_content"):
            return []
        
        formatted_content = search_results.get("formatted_content", {})
        
        if not formatted_content.get("sources"):
            return []
        
        citations = []
        
        for source in formatted_content["sources"]:
            url = source.get("url", "")
            title = source.get("title", "Unknown Source")
            
            if citation_style == "academic":
                # Academic citation style
                citations.append(f'"{title}". Retrieved from {url}')
            elif citation_style == "minimal":
                # Minimal citation style
                citations.append(url)
            else:
                # Standard citation style
                citations.append(f"{title} - {url}")
        
        return citations
    
    def resolve_conflicts(self, llm_output: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts between LLM output and search results.
        
        Args:
            llm_output: Output from the LLM
            search_results: Formatted search results
            
        Returns:
            Dictionary with conflict resolution information
        """
        if not search_results.get("success", False) or not search_results.get("formatted_content"):
            return {
                "has_conflicts": False,
                "resolved_output": llm_output,
                "confidence": "high"
            }
        
        formatted_content = search_results.get("formatted_content", {})
        
        # Extract key facts from LLM output
        llm_facts = self._extract_facts(llm_output)
        
        # Extract key facts from search results
        search_facts = self._extract_facts(formatted_content.get("context", ""))
        
        # Compare facts to find conflicts
        conflicts = []
        for llm_fact in llm_facts:
            contradicted = False
            contradicting_fact = ""
            
            for search_fact in search_facts:
                if self._are_contradictory(llm_fact, search_fact):
                    contradicted = True
                    contradicting_fact = search_fact
                    break
            
            if contradicted:
                conflicts.append({
                    "llm_fact": llm_fact,
                    "search_fact": contradicting_fact
                })
        
        # If no conflicts, return original output
        if not conflicts:
            return {
                "has_conflicts": False,
                "resolved_output": llm_output,
                "confidence": "high"
            }
        
        # Attempt to resolve conflicts
        resolved_output = llm_output
        
        for conflict in conflicts:
            # Replace contradicted fact with search fact
            llm_fact = conflict["llm_fact"]
            search_fact = conflict["search_fact"]
            
            # Simple substitution (for more complex cases, would need more sophisticated NLP)
            resolved_output = resolved_output.replace(llm_fact, search_fact)
        
        return {
            "has_conflicts": True,
            "conflicts": conflicts,
            "resolved_output": resolved_output,
            "confidence": "medium"
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """
        Extract factual statements from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted facts
        """
        # Basic fact extraction (for more sophisticated extraction, would need NLP)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        facts = []
        for sentence in sentences:
            # Filter out non-factual sentences
            if (
                len(sentence) > 10 and  # Reasonably long
                not sentence.startswith("I ") and  # Not a personal statement
                not "might" in sentence.lower() and  # Not speculative
                not "may" in sentence.lower() and
                not "could" in sentence.lower() and
                not "perhaps" in sentence.lower() and
                not "possibly" in sentence.lower() and
                not "?" in sentence  # Not a question
            ):
                facts.append(sentence)
        
        return facts
    
    def _are_contradictory(self, fact1: str, fact2: str) -> bool:
        """
        Check if two facts contradict each other.
        
        Args:
            fact1: First fact
            fact2: Second fact
            
        Returns:
            True if facts contradict, False otherwise
        """
        # This is a simplified implementation
        # A more sophisticated approach would use NLP to identify contradictions
        
        # Look for direct negation
        negation_patterns = [
            (r'is not', r'is'),
            (r'is', r'is not'),
            (r'are not', r'are'),
            (r'are', r'are not'),
            (r'does not', r'does'),
            (r'does', r'does not'),
            (r'has not', r'has'),
            (r'has', r'has not')
        ]
        
        for pattern1, pattern2 in negation_patterns:
            if re.search(pattern1, fact1) and re.search(pattern2, fact2):
                # Extract the context around the negation
                context1 = re.search(f'{pattern1} (.*?)(?:[.!?]|$)', fact1)
                context2 = re.search(f'{pattern2} (.*?)(?:[.!?]|$)', fact2)
                
                if context1 and context2:
                    # Check if the contexts are similar
                    similarity = self._text_similarity(context1.group(1), context2.group(1))
                    if similarity > 0.7:  # Threshold for similarity
                        return True
        
        # Look for numerical contradictions
        # Extract numbers from both facts
        numbers1 = re.findall(r'\b\d+(?:\.\d+)?\b', fact1)
        numbers2 = re.findall(r'\b\d+(?:\.\d+)?\b', fact2)
        
        # If both facts contain numbers
        if numbers1 and numbers2:
            # Convert to float for comparison
            float_numbers1 = [float(n) for n in numbers1]
            float_numbers2 = [float(n) for n in numbers2]
            
            # Compare numbers with context
            for n1 in float_numbers1:
                for n2 in float_numbers2:
                    # If numbers are significantly different
                    if abs(n1 - n2) / max(n1, n2) > 0.1:  # 10% difference threshold
                        # Check if they appear in similar contexts
                        context1 = re.search(f'(.*?){n1}(.*?)(?:[.!?]|$)', fact1)
                        context2 = re.search(f'(.*?){n2}(.*?)(?:[.!?]|$)', fact2)
                        
                        if context1 and context2:
                            pre_similarity = self._text_similarity(context1.group(1), context2.group(1))
                            post_similarity = self._text_similarity(context1.group(2), context2.group(2))
                            
                            if pre_similarity > 0.6 and post_similarity > 0.6:
                                return True
        
        # Look for contradictory mathematical operations
        math_operations = [
            (r'greater than', r'less than'),
            (r'increases', r'decreases'),
            (r'positive', r'negative'),
            (r'maximum', r'minimum'),
            (r'converges', r'diverges'),
            (r'continuous', r'discontinuous')
        ]
        
        for op1, op2 in math_operations:
            if re.search(op1, fact1) and re.search(op2, fact2):
                # Check for similar context
                context1 = re.search(f'(.*?){op1}(.*?)(?:[.!?]|$)', fact1)
                context2 = re.search(f'(.*?){op2}(.*?)(?:[.!?]|$)', fact2)
                
                if context1 and context2:
                    pre_similarity = self._text_similarity(context1.group(1), context2.group(1))
                    post_similarity = self._text_similarity(context1.group(2), context2.group(2))
                    
                    if pre_similarity > 0.6 and post_similarity > 0.6:
                        return True
        
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Split into words
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def format_response_with_citations(self, llm_output: str, search_results: Dict[str, Any]) -> str:
        """
        Format LLM output with citations from search results.
        
        Args:
            llm_output: Original LLM output
            search_results: Formatted search results
            
        Returns:
            Output with added citations
        """
        if not search_results.get("success", False) or not search_results.get("formatted_content"):
            return llm_output
        
        formatted_content = search_results.get("formatted_content", {})
        
        if not formatted_content.get("sources"):
            return llm_output
        
        # Generate citations
        citations = self.generate_citations(search_results, "standard")
        
        if not citations:
            return llm_output
        
        # Add citation section
        citation_text = "\n\nSources:\n"
        for i, citation in enumerate(citations, 1):
            citation_text += f"{i}. {citation}\n"
        
        return llm_output + citation_text
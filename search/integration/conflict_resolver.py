"""
Conflict resolution between LLM output and search results.

This module provides specialized functionality to identify and resolve conflicts
between the LLM's generated output and information retrieved from external sources.
"""

import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ConflictResolver:
    """
    Resolves conflicts between LLM output and search results.
    
    This class identifies potential contradictions between information generated
    by the LLM and information retrieved from external sources, and attempts to
    resolve these conflicts in favor of more authoritative information.
    """
    
    def __init__(self):
        """Initialize the conflict resolver."""
        # Confidence thresholds for different types of information
        self.thresholds = {
            "high_credibility": 0.8,  # Threshold for high credibility sources
            "medium_credibility": 0.6,  # Threshold for medium credibility sources
            "mathematical_facts": 0.9,  # Threshold for well-established mathematical facts
            "numerical_values": 0.7,  # Threshold for numerical values
            "definitions": 0.85  # Threshold for mathematical definitions
        }
    
    def resolve_conflicts(self, llm_output: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify and resolve conflicts between LLM output and search results.
        
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
        
        # Extract mathematical statements from LLM output
        llm_statements = self._extract_mathematical_statements(llm_output)
        
        # Extract mathematical statements from search results
        search_statements = self._extract_mathematical_statements(formatted_content.get("context", ""))
        
        # Identify potential conflicts
        conflicts = self._identify_conflicts(llm_statements, search_statements)
        
        # If no conflicts, return original output
        if not conflicts:
            return {
                "has_conflicts": False,
                "resolved_output": llm_output,
                "confidence": "high"
            }
        
        # Resolve conflicts
        resolved_output = llm_output
        resolved_conflicts = []
        
        for conflict in conflicts:
            llm_statement = conflict["llm_statement"]
            search_statement = conflict["search_statement"]
            confidence = conflict["confidence"]
            
            # If confidence exceeds threshold, replace LLM statement with search statement
            if confidence > self.thresholds.get(conflict["type"], 0.7):
                resolved_output = self._replace_statement(resolved_output, llm_statement, search_statement)
                resolved_conflicts.append({
                    "original": llm_statement,
                    "replacement": search_statement,
                    "confidence": confidence,
                    "resolved": True
                })
            else:
                # Otherwise, mark as unresolved
                resolved_conflicts.append({
                    "original": llm_statement,
                    "alternative": search_statement,
                    "confidence": confidence,
                    "resolved": False
                })
        
        # Determine overall confidence
        if all(conflict["resolved"] for conflict in resolved_conflicts):
            overall_confidence = "high"
        elif any(conflict["resolved"] for conflict in resolved_conflicts):
            overall_confidence = "medium"
        else:
            overall_confidence = "low"
        
        return {
            "has_conflicts": True,
            "conflicts": resolved_conflicts,
            "resolved_output": resolved_output,
            "confidence": overall_confidence
        }
    
    def _extract_mathematical_statements(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical statements from text.
        
        Args:
            text: Input text
            
        Returns:
            List of mathematical statement dictionaries
        """
        statements = []
        
        # Extract complete sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # Skip short or non-statement sentences
            if len(sentence) < 10 or "?" in sentence:
                continue
            
            # Identify statement type
            statement_type = self._identify_statement_type(sentence)
            
            if statement_type:
                statements.append({
                    "text": sentence,
                    "type": statement_type
                })
        
        return statements
    
    def _identify_statement_type(self, sentence: str) -> Optional[str]:
        """
        Identify the type of mathematical statement.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Statement type or None if not a mathematical statement
        """
        sentence_lower = sentence.lower()
        
        # Check for definitions
        if any(pattern in sentence_lower for pattern in ["is defined as", "is the", "refers to", "means", "definition"]):
            return "definitions"
        
        # Check for numerical statements
        if re.search(r'\b\d+(?:\.\d+)?\b', sentence):
            return "numerical_values"
        
        # Check for mathematical facts
        math_terms = [
            "theorem", "prove", "proof", "equals", "equal to", "equivalent", "formula",
            "equation", "identity", "property", "rule", "always", "never", "converges",
            "diverges", "increases", "decreases", "maximum", "minimum"
        ]
        
        if any(term in sentence_lower for term in math_terms):
            return "mathematical_facts"
        
        # Check for general mathematical statements
        math_symbols = ["+", "-", "*", "/", "=", "<", ">", "≤", "≥", "≠", "∈", "∉", "⊂", "⊃", "∩", "∪", "∫", "∑", "∏", "→", "⇒", "⟹"]
        
        if any(symbol in sentence for symbol in math_symbols) or "$" in sentence:
            return "mathematical_facts"
        
        return None
    
    def _identify_conflicts(self, llm_statements: List[Dict[str, Any]], search_statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify conflicts between LLM statements and search statements.
        
        Args:
            llm_statements: Statements from LLM output
            search_statements: Statements from search results
            
        Returns:
            List of conflict dictionaries
        """
        conflicts = []
        
        for llm_statement in llm_statements:
            llm_text = llm_statement["text"]
            llm_type = llm_statement["type"]
            
            for search_statement in search_statements:
                search_text = search_statement["text"]
                search_type = search_statement["type"]
                
                # Only compare statements of the same type
                if llm_type == search_type:
                    # Calculate conflict score
                    conflict_score, conflict_type = self._calculate_conflict_score(llm_text, search_text, llm_type)
                    
                    # If conflict score is significant
                    if conflict_score > 0.5:
                        conflicts.append({
                            "llm_statement": llm_text,
                            "search_statement": search_text,
                            "type": conflict_type,
                            "confidence": conflict_score
                        })
        
        return conflicts
    
    def _calculate_conflict_score(self, statement1: str, statement2: str, statement_type: str) -> tuple:
        """
        Calculate conflict score between two statements.
        
        Args:
            statement1: First statement
            statement2: Second statement
            statement_type: Type of statement
            
        Returns:
            Tuple of (conflict score, conflict type)
        """
        # Normalize statements
        statement1 = statement1.lower()
        statement2 = statement2.lower()
        
        # Calculate similarity to ensure we're comparing related statements
        similarity = self._calculate_similarity(statement1, statement2)
        
        # If statements are not similar enough, they're not in conflict
        if similarity < 0.3:
            return 0.0, ""
        
        conflict_score = 0.0
        conflict_type = ""
        
        if statement_type == "numerical_values":
            # Extract numbers
            numbers1 = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', statement1)]
            numbers2 = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', statement2)]
            
            # Compare numbers
            if numbers1 and numbers2:
                # Calculate relative difference
                diffs = [abs(n1 - n2) / max(abs(n1), abs(n2), 1e-10) for n1 in numbers1 for n2 in numbers2]
                
                if diffs:
                    min_diff = min(diffs)
                    # Convert difference to conflict score
                    if min_diff > 0.01:  # 1% difference threshold
                        conflict_score = min(1.0, min_diff * 10)  # Scale up to 1.0
                        conflict_type = "numerical_values"
        
        elif statement_type == "definitions":
            # Check for contradictory definitions
            negation_patterns = [
                (r'is not', r'is'),
                (r'is', r'is not'),
                (r'are not', r'are'),
                (r'are', r'are not'),
                (r'does not', r'does'),
                (r'does', r'does not')
            ]
            
            for pattern1, pattern2 in negation_patterns:
                if re.search(f'\\b{pattern1}\\b', statement1) and re.search(f'\\b{pattern2}\\b', statement2):
                    conflict_score = 0.9
                    conflict_type = "definitions"
                    break
        
        elif statement_type == "mathematical_facts":
            # Check for contradictory statements
            contradictions = [
                (r'increases', r'decreases'),
                (r'positive', r'negative'),
                (r'greater than', r'less than'),
                (r'maximum', r'minimum'),
                (r'converges', r'diverges'),
                (r'always', r'never'),
                (r'equal to', r'not equal to'),
                (r'equivalent', r'not equivalent')
            ]
            
            for term1, term2 in contradictions:
                if re.search(f'\\b{term1}\\b', statement1) and re.search(f'\\b{term2}\\b', statement2):
                    conflict_score = 0.8
                    conflict_type = "mathematical_facts"
                    break
            
            # Check for symbol contradictions
            if "$" in statement1 and "$" in statement2:
                # Extract LaTeX expressions
                latex1 = re.findall(r'\$(.*?)\$', statement1)
                latex2 = re.findall(r'\$(.*?)\$', statement2)
                
                if latex1 and latex2:
                    # Simple check for different expressions in similar contexts
                    if latex1[0] != latex2[0]:
                        pre1 = statement1.split("$")[0] if "$" in statement1 else ""
                        pre2 = statement2.split("$")[0] if "$" in statement2 else ""
                        
                        pre_similarity = self._calculate_similarity(pre1, pre2)
                        
                        if pre_similarity > 0.7:
                            conflict_score = 0.7
                            conflict_type = "mathematical_facts"
        
        # If credibility information is available, adjust conflict score
        credibility_adjustment = self._get_credibility_adjustment(statement2)
        conflict_score *= credibility_adjustment
        
        return conflict_score, conflict_type or statement_type
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Remove punctuation and normalize whitespace
        text1 = re.sub(r'[^\w\s]', '', text1).lower()
        text2 = re.sub(r'[^\w\s]', '', text2).lower()
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "of", "to", "for", "with", "by", "is", "are", "was", "were"}
        words1 = words1.difference(stop_words)
        words2 = words2.difference(stop_words)
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def _get_credibility_adjustment(self, statement: str) -> float:
        """
        Get credibility adjustment factor for a statement.
        
        Args:
            statement: Statement to evaluate
            
        Returns:
            Credibility adjustment factor (0.5 to 1.2)
        """
        # This is a placeholder - in a real implementation, this would assess
        # the credibility of the source of the statement
        
        # Default credibility
        return 1.0
    
    def _replace_statement(self, text: str, old_statement: str, new_statement: str) -> str:
        """
        Replace a statement in text.
        
        Args:
            text: Original text
            old_statement: Statement to replace
            new_statement: Replacement statement
            
        Returns:
            Updated text
        """
        # Simple direct replacement
        # In a more sophisticated implementation, this would ensure grammatical correctness
        return text.replace(old_statement, new_statement)
"""
Response Formatter Module

This module provides utilities for formatting comprehensive mathematical responses
that integrate text explanations, LaTeX expressions, step-by-step solutions,
and visualizations into a unified, coherent presentation.
"""

import logging
import re
from typing import Dict, List, Optional, Union, Any

from math_processing.formatting.latex_formatter import LatexFormatter

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Formats comprehensive mathematical responses with integrated content types.
    
    This class handles the integration of natural language explanations, 
    LaTeX expressions, step-by-step solutions, and visualizations into
    a coherent, well-structured response that adapts to the complexity
    of the mathematical content and user needs.
    """
    
    def __init__(self, latex_formatter: Optional[LatexFormatter] = None):
        """
        Initialize the response formatter.
        
        Args:
            latex_formatter: Optional LaTeX formatter for mathematical expressions
        """
        self.latex_formatter = latex_formatter or LatexFormatter()
        
    def format_response(self, 
                       response_data: Dict[str, Any],
                       format_type: str = "default",
                       complexity_level: str = "auto",
                       include_citations: bool = True) -> Dict[str, Any]:
        """
        Format a comprehensive mathematical response.
        
        Args:
            response_data: Dictionary containing response components
            format_type: Response format type ('default', 'educational', 'concise', 'technical')
            complexity_level: Detail level ('basic', 'intermediate', 'advanced', 'auto')
            include_citations: Whether to include citations for external sources
            
        Returns:
            Formatted response dictionary
        """
        # Extract response components
        explanation = response_data.get("explanation", "")
        latex_expressions = response_data.get("latex_expressions", [])
        steps = response_data.get("steps", [])
        visualizations = response_data.get("visualizations", [])
        citations = response_data.get("citations", [])
        
        # Detect complexity if set to auto
        if complexity_level == "auto":
            complexity_level = self._detect_complexity(response_data)
        
        # Format based on response type
        if format_type == "educational":
            formatted_response = self._format_educational_response(
                explanation, latex_expressions, steps, visualizations, citations,
                complexity_level, include_citations
            )
        elif format_type == "concise":
            formatted_response = self._format_concise_response(
                explanation, latex_expressions, steps, visualizations, citations,
                complexity_level, include_citations
            )
        elif format_type == "technical":
            formatted_response = self._format_technical_response(
                explanation, latex_expressions, steps, visualizations, citations,
                complexity_level, include_citations
            )
        else:  # default format
            formatted_response = self._format_default_response(
                explanation, latex_expressions, steps, visualizations, citations,
                complexity_level, include_citations
            )
            
        return formatted_response
        
    def _detect_complexity(self, response_data: Dict[str, Any]) -> str:
        """
        Detect the complexity level of the mathematical content.
        
        Args:
            response_data: Dictionary containing response components
            
        Returns:
            Complexity level ('basic', 'intermediate', 'advanced')
        """
        # Extract components for analysis
        explanation = response_data.get("explanation", "")
        latex_expressions = response_data.get("latex_expressions", [])
        steps = response_data.get("steps", [])
        domain = response_data.get("domain", "")
        
        # Initialize complexity score
        complexity_score = 0
        
        # Check explanation complexity
        word_count = len(explanation.split())
        if word_count > 300:
            complexity_score += 2
        elif word_count > 150:
            complexity_score += 1
            
        # Check LaTeX complexity
        if isinstance(latex_expressions, list):
            for expr in latex_expressions:
                complexity_score += self._get_expression_complexity(expr)
        elif isinstance(latex_expressions, str):
            complexity_score += self._get_expression_complexity(latex_expressions)
            
        # Check solution steps complexity
        if steps:
            complexity_score += min(len(steps) // 3, 3)  # Cap at 3 points
            
            # Check if steps contain advanced operations
            advanced_operations = ["integrate", "differentiate", "limit", "series"]
            for step in steps:
                operation = step.get("operation", "")
                if operation in advanced_operations:
                    complexity_score += 1
                    break
                    
        # Domain-specific complexity
        advanced_domains = ["calculus", "linear_algebra", "differential_equations"]
        if domain in advanced_domains:
            complexity_score += 1
            
        # Determine complexity level
        if complexity_score <= 2:
            return "basic"
        elif complexity_score <= 5:
            return "intermediate"
        else:
            return "advanced"
    
    def _get_expression_complexity(self, latex_expr: str) -> int:
        """
        Calculate the complexity score of a LaTeX expression.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Complexity score (0-5)
        """
        complexity = 0
        
        # Check for complex structures
        if "\\frac" in latex_expr:
            complexity += 1
        if "\\sum" in latex_expr or "\\prod" in latex_expr or "\\int" in latex_expr:
            complexity += 1
        if "\\lim" in latex_expr:
            complexity += 1
        if "matrix" in latex_expr:
            complexity += 2
            
        # Check for nested structures
        nested_level = 0
        for char in latex_expr:
            if char == '{':
                nested_level += 1
            elif char == '}':
                nested_level = max(0, nested_level - 1)
                
        # Maximum nesting level as an indicator of complexity
        complexity += min(nested_level // 2, 2)  # Cap at 2 points
        
        return min(complexity, 5)  # Cap at 5 points
        
    def _format_default_response(self, 
                                explanation: str,
                                latex_expressions: Union[List[str], str],
                                steps: List[Dict[str, Any]],
                                visualizations: List[Dict[str, Any]],
                                citations: List[str],
                                complexity_level: str,
                                include_citations: bool) -> Dict[str, Any]:
        """
        Format a default response with balanced content.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            visualizations: Visualization metadata
            citations: Citation information
            complexity_level: Complexity level
            include_citations: Whether to include citations
            
        Returns:
            Formatted response dictionary
        """
        # Format LaTeX expressions
        formatted_latex = []
        if isinstance(latex_expressions, list):
            for expr in latex_expressions:
                formatted_latex.append(self.latex_formatter.format_expression(expr))
        elif isinstance(latex_expressions, str):
            formatted_latex = self.latex_formatter.format_expression(latex_expressions)
            
        # Format solution steps
        formatted_steps = []
        if steps:
            formatted_steps = self.latex_formatter.format_step_solution(steps)
            
        # Structure visualizations
        structured_visualizations = self._structure_visualizations(visualizations, complexity_level)
            
        # Format explanation based on complexity
        formatted_explanation = self._format_explanation(explanation, complexity_level)
        
        # Add math expressions to explanation if needed
        if formatted_latex and isinstance(formatted_latex, str):
            if "\\begin{equation}" not in formatted_explanation and "$$" not in formatted_explanation:
                display_expr = self.latex_formatter.create_display_math(formatted_latex)
                formatted_explanation += f"\n\n{display_expr}\n\n"
        
        # Prepare citation text if needed
        citation_text = ""
        if include_citations and citations:
            citation_text = self._format_citations(citations)
            
        # Assemble the final response
        formatted_response = {
            "explanation": formatted_explanation,
            "latex_expressions": formatted_latex,
            "steps": formatted_steps,
            "visualizations": structured_visualizations,
            "citation_text": citation_text,
            "complexity_level": complexity_level,
            "format_type": "default"
        }
        
        return formatted_response
        
    def _format_educational_response(self, 
                                    explanation: str,
                                    latex_expressions: Union[List[str], str],
                                    steps: List[Dict[str, Any]],
                                    visualizations: List[Dict[str, Any]],
                                    citations: List[str],
                                    complexity_level: str,
                                    include_citations: bool) -> Dict[str, Any]:
        """
        Format an educational response with detailed explanations and examples.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            visualizations: Visualization metadata
            citations: Citation information
            complexity_level: Complexity level
            include_citations: Whether to include citations
            
        Returns:
            Formatted response dictionary
        """
        # Format LaTeX expressions
        formatted_latex = []
        if isinstance(latex_expressions, list):
            for expr in latex_expressions:
                formatted_latex.append(self.latex_formatter.format_expression(expr))
        elif isinstance(latex_expressions, str):
            formatted_latex = self.latex_formatter.format_expression(latex_expressions)
            
        # Format solution steps with additional educational notes
        formatted_steps = []
        if steps:
            # For educational format, add learning notes to steps
            enhanced_steps = self._enhance_steps_with_educational_notes(steps)
            formatted_steps = self.latex_formatter.format_step_solution(enhanced_steps)
            
        # Structure visualizations with educational context
        structured_visualizations = self._structure_visualizations(
            visualizations, complexity_level, educational=True
        )
            
        # Format extended explanation with foundational concepts
        formatted_explanation = self._format_educational_explanation(explanation, complexity_level)
        
        # Add conceptual breakdown section
        conceptual_breakdown = self._create_conceptual_breakdown(
            explanation, latex_expressions, steps, complexity_level
        )
        
        # Prepare citation text if needed
        citation_text = ""
        if include_citations and citations:
            citation_text = self._format_citations(citations)
            
        # Add learning objectives based on complexity
        learning_objectives = self._generate_learning_objectives(
            explanation, latex_expressions, steps, complexity_level
        )
            
        # Assemble the final response
        formatted_response = {
            "explanation": formatted_explanation,
            "latex_expressions": formatted_latex,
            "steps": formatted_steps,
            "visualizations": structured_visualizations,
            "conceptual_breakdown": conceptual_breakdown,
            "learning_objectives": learning_objectives,
            "citation_text": citation_text,
            "complexity_level": complexity_level,
            "format_type": "educational"
        }
        
        return formatted_response
        
    def _format_concise_response(self, 
                                explanation: str,
                                latex_expressions: Union[List[str], str],
                                steps: List[Dict[str, Any]],
                                visualizations: List[Dict[str, Any]],
                                citations: List[str],
                                complexity_level: str,
                                include_citations: bool) -> Dict[str, Any]:
        """
        Format a concise response with essential information only.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            visualizations: Visualization metadata
            citations: Citation information
            complexity_level: Complexity level
            include_citations: Whether to include citations
            
        Returns:
            Formatted response dictionary
        """
        # Format LaTeX expressions
        formatted_latex = []
        if isinstance(latex_expressions, list):
            for expr in latex_expressions:
                formatted_latex.append(self.latex_formatter.format_expression(expr))
        elif isinstance(latex_expressions, str):
            formatted_latex = self.latex_formatter.format_expression(latex_expressions)
            
        # For concise format, only include key steps
        formatted_steps = []
        if steps:
            # Extract only essential steps (first, key intermediate, and last)
            essential_steps = self._extract_essential_steps(steps)
            formatted_steps = self.latex_formatter.format_step_solution(essential_steps)
            
        # For concise format, include at most one visualization
        structured_visualizations = []
        if visualizations:
            # Pick the most informative visualization
            best_viz = self._select_best_visualization(visualizations)
            if best_viz:
                structured_visualizations = [best_viz]
                
        # Create a concise explanation
        formatted_explanation = self._create_concise_explanation(explanation)
        
        # Prepare minimal citation text if needed
        citation_text = ""
        if include_citations and citations:
            citation_text = self._format_minimal_citations(citations)
            
        # Assemble the final response
        formatted_response = {
            "explanation": formatted_explanation,
            "latex_expressions": formatted_latex,
            "steps": formatted_steps,
            "visualizations": structured_visualizations,
            "citation_text": citation_text,
            "complexity_level": complexity_level,
            "format_type": "concise"
        }
        
        return formatted_response
        
    def _format_technical_response(self, 
                                  explanation: str,
                                  latex_expressions: Union[List[str], str],
                                  steps: List[Dict[str, Any]],
                                  visualizations: List[Dict[str, Any]],
                                  citations: List[str],
                                  complexity_level: str,
                                  include_citations: bool) -> Dict[str, Any]:
        """
        Format a technical response with formal mathematical presentation.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            visualizations: Visualization metadata
            citations: Citation information
            complexity_level: Complexity level
            include_citations: Whether to include citations
            
        Returns:
            Formatted response dictionary
        """
        # Format LaTeX expressions with strict mathematical formatting
        formatted_latex = []
        if isinstance(latex_expressions, list):
            for expr in latex_expressions:
                formatted_latex.append(
                    self.latex_formatter.format_expression(expr)
                )
        elif isinstance(latex_expressions, str):
            formatted_latex = self.latex_formatter.format_expression(latex_expressions)
            
        # Format solution steps with formal mathematical notation
        formatted_steps = []
        if steps:
            formatted_steps = self.latex_formatter.format_step_solution(steps)
            
        # Structure visualizations with technical details
        structured_visualizations = self._structure_visualizations(
            visualizations, complexity_level, technical=True
        )
            
        # Format explanation with formal mathematical language
        formatted_explanation = self._format_technical_explanation(explanation, complexity_level)
        
        # Add formal mathematical notation using equation environments
        if formatted_latex:
            if isinstance(formatted_latex, list):
                equation_envs = self.latex_formatter.create_equation_environment(
                    formatted_latex, numbered=True, aligned=True
                )
            else:
                equation_envs = self.latex_formatter.create_equation_environment(
                    formatted_latex, numbered=True
                )
            formatted_explanation += f"\n\n{equation_envs}\n\n"
            
        # Prepare formal citations if needed
        citation_text = ""
        if include_citations and citations:
            citation_text = self._format_formal_citations(citations)
            
        # Add theoretical context section
        theoretical_context = self._create_theoretical_context(
            explanation, latex_expressions, steps, complexity_level
        )
            
        # Assemble the final response
        formatted_response = {
            "explanation": formatted_explanation,
            "latex_expressions": formatted_latex,
            "steps": formatted_steps,
            "visualizations": structured_visualizations,
            "theoretical_context": theoretical_context,
            "citation_text": citation_text,
            "complexity_level": complexity_level,
            "format_type": "technical"
        }
        
        return formatted_response
    
    def _format_explanation(self, explanation: str, complexity_level: str) -> str:
        """
        Format an explanation based on complexity level.
        
        Args:
            explanation: Original explanation
            complexity_level: Complexity level
            
        Returns:
            Formatted explanation
        """
        if not explanation:
            return ""
            
        # Basic adjustments for all complexity levels
        formatted_text = explanation
        
        # Replace plain text math with inline LaTeX
        formatted_text = self._convert_plain_math_to_latex(formatted_text)
        
        # Adjust based on complexity level
        if complexity_level == "basic":
            # Simplify language for basic level
            # Replace complex terms with simpler ones
            complex_terms = {
                r'\b(derivative)\b': 'rate of change',
                r'\b(integral)\b': 'area under the curve',
                r'\b(orthogonal)\b': 'perpendicular',
                r'\b(asymptote)\b': 'line that the curve approaches',
                # Add more term replacements as needed
            }
            
            for term, replacement in complex_terms.items():
                formatted_text = re.sub(
                    term, 
                    f"{replacement} (technically called {term.strip('\\b')})", 
                    formatted_text, 
                    flags=re.IGNORECASE
                )
                
        elif complexity_level == "advanced":
            # For advanced level, ensure proper mathematical terminology
            simplified_terms = {
                r'\b(rate of change)\b': 'derivative',
                r'\b(area under the curve)\b': 'integral',
                r'\b(perpendicular)\b': 'orthogonal',
                # Add more term replacements as needed
            }
            
            for term, replacement in simplified_terms.items():
                formatted_text = re.sub(term, replacement, formatted_text, flags=re.IGNORECASE)
                
        return formatted_text
        
    def _format_educational_explanation(self, explanation: str, complexity_level: str) -> str:
        """
        Format an explanation for educational purposes with additional context.
        
        Args:
            explanation: Original explanation
            complexity_level: Complexity level
            
        Returns:
            Educational explanation
        """
        # First apply basic formatting
        formatted_text = self._format_explanation(explanation, complexity_level)
        
        # Add educational enhancements
        # Highlight key terms with emphasis
        key_math_terms = [
            "derivative", "integral", "function", "equation", "variable",
            "theorem", "proof", "formula", "coefficient", "expression",
            "matrix", "vector", "scalar", "limit", "series", "sequence",
            "convergence", "divergence", "factorial", "permutation",
            "combination", "probability", "statistics", "hypothesis"
            # Add more terms as needed
        ]
        
        for term in key_math_terms:
            # Only highlight the first occurrence of each term
            pattern = f"\\b({term})\\b"
            match = re.search(pattern, formatted_text, re.IGNORECASE)
            if match:
                replacement = f"**{match.group(1)}**"  # Bold formatting
                formatted_text = formatted_text[:match.start()] + replacement + formatted_text[match.end():]
        
        # Add educational headers if not present
        if not any(header in formatted_text for header in ["# ", "## ", "### "]):
            formatted_text = "## Mathematical Explanation\n\n" + formatted_text
            
        # Add "Understanding the Concept" section if not too long already
        if len(formatted_text.split()) < 300:
            conceptual_note = (
                "\n\n### Understanding the Concept\n\n"
                "To better understand this mathematical concept, think of it in terms "
                "of real-world applications. Mathematical ideas often represent patterns "
                "and relationships that exist in the physical world, and visualizing these "
                "connections can help solidify your understanding."
            )
            formatted_text += conceptual_note
            
        return formatted_text
        
    def _create_concise_explanation(self, explanation: str) -> str:
        """
        Create a concise version of an explanation.
        
        Args:
            explanation: Original explanation
            
        Returns:
            Concise explanation
        """
        if not explanation:
            return ""
            
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', explanation)
        
        # For very short explanations, just return as is
        if len(sentences) <= 3:
            return explanation
            
        # Extract key sentences (first, possibly middle, and last)
        key_sentences = [sentences[0]]  # Always include first sentence
        
        # Include an important middle sentence if explanation is long enough
        if len(sentences) >= 5:
            # Look for sentences with key indicators of importance
            importance_indicators = [
                "therefore", "thus", "hence", "consequently", "as a result",
                "means that", "implies", "shows", "proves", "demonstrates",
                "key", "important", "essential", "critical", "significant"
            ]
            
            for i in range(1, len(sentences) - 1):
                if any(indicator in sentences[i].lower() for indicator in importance_indicators):
                    key_sentences.append(sentences[i])
                    break
                    
        # Always include final sentence
        if len(sentences) > 1:
            key_sentences.append(sentences[-1])
            
        # Join the key sentences
        concise_explanation = " ".join(key_sentences)
        
        return concise_explanation
        
    def _format_technical_explanation(self, explanation: str, complexity_level: str) -> str:
        """
        Format an explanation with formal mathematical language.
        
        Args:
            explanation: Original explanation
            complexity_level: Complexity level
            
        Returns:
            Technical explanation
        """
        # First apply basic formatting
        formatted_text = self._format_explanation(explanation, complexity_level)
        
        # Apply technical enhancements
        # Replace informal expressions with formal ones
        informal_to_formal = {
            r'\b(we can see that)\b': 'it follows that',
            r'\b(we find that)\b': 'we obtain',
            r'\b(plugging in)\b': 'substituting',
            r'\b(cancel out)\b': 'eliminate',
            r'\b(work out)\b': 'compute',
            r'\b(figure out)\b': 'determine',
            r'\b(use)\b': 'apply',
            r'\b(put)\b': 'substitute',
            r'\b(get)\b': 'obtain',
            # Add more replacements as needed
        }
        
        for informal, formal in informal_to_formal.items():
            formatted_text = re.sub(informal, formal, formatted_text, flags=re.IGNORECASE)
            
        # Ensure proper mathematical formatting
        # Replace word-based operators with symbolic representations
        word_to_symbol = {
            r'\b(multiplied by)\b': '$\\cdot$',
            r'\b(times)\b': '$\\times$',
            r'\b(divided by)\b': '$/$',
            r'\b(is equal to)\b': '$=$',
            r'\b(equals)\b': '$=$',
            r'\b(is less than)\b': '$<$',
            r'\b(is greater than)\b': '$>$',
            r'\b(is less than or equal to)\b': '$\\leq$',
            r'\b(is greater than or equal to)\b': '$\\geq$',
            # Add more replacements as needed
        }
        
        for word, symbol in word_to_symbol.items():
            formatted_text = re.sub(word, symbol, formatted_text, flags=re.IGNORECASE)
            
        return formatted_text
        
    def _convert_plain_math_to_latex(self, text: str) -> str:
        """
        Convert plain text mathematical expressions to LaTeX.
        
        Args:
            text: Text containing plain math expressions
            
        Returns:
            Text with LaTeX formatting
        """
        if not text:
            return ""
            
        # Common patterns to replace with LaTeX
        patterns = [
            # Fractions like x/y
            (r'(\b\d+)/(\d+\b)', r'$\\frac{\1}{\2}$'),
            
            # Exponents like x^2
            (r'(\b[a-zA-Z0-9]+)\^(\d+\b)', r'$\1^{\2}$'),
            
            # Square roots like sqrt(x)
            (r'sqrt\(([^)]+)\)', r'$\\sqrt{\1}$'),
            
            # Common mathematical functions
            (r'\bsin\(([^)]+)\)', r'$\\sin(\1)$'),
            (r'\bcos\(([^)]+)\)', r'$\\cos(\1)$'),
            (r'\btan\(([^)]+)\)', r'$\\tan(\1)$'),
            (r'\bln\(([^)]+)\)', r'$\\ln(\1)$'),
            (r'\blog\(([^)]+)\)', r'$\\log(\1)$'),
            
            # Integrals, derivatives, limits (simple forms)
            (r'\bintegral of ([^d]+) dx\b', r'$\\int \1 \\, dx$'),
            (r'\bderivative of ([^w]+) with respect to ([a-z])\b', r'$\\frac{d\1}{d\2}$'),
            (r'\blimit of ([^a]+) as ([a-z]) approaches ([^i]+)\b', r'$\\lim_{\2 \\to \3} \1$'),
            
            # Add more patterns as needed
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
            
        return text
        
    def _enhance_steps_with_educational_notes(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance solution steps with educational notes.
        
        Args:
            steps: Original solution steps
            
        Returns:
            Steps with educational enhancements
        """
        if not steps:
            return []
            
        enhanced_steps = []
        
        for i, step in enumerate(steps):
            enhanced_step = step.copy()
            
            # Add educational note based on operation type
            operation = step.get("operation", "")
            explanation = step.get("explanation", "")
            
            educational_note = ""
            
            if operation == "differentiate":
                educational_note = (
                    "Note: Differentiation measures the rate of change of a function. "
                    "It tells us how quickly a function's output changes as we vary the input."
                )
            elif operation == "integrate":
                educational_note = (
                    "Note: Integration can be thought of as finding the area under a curve. "
                    "It's the reverse process of differentiation."
                )
            elif operation == "substitute":
                educational_note = (
                    "Note: Substitution allows us to replace variables or expressions with their values, "
                    "simplifying the problem or preparing for the next step."
                )
            elif operation == "solve":
                educational_note = (
                    "Note: When solving equations, we isolate the variable by performing "
                    "the same operations on both sides to maintain equality."
                )
            elif operation == "factor":
                educational_note = (
                    "Note: Factoring breaks down an expression into simpler parts, "
                    "often revealing important information about the expression."
                )
            # Add more operation types as needed
            
            # Add the note if we have one and it's not redundant with the explanation
            if educational_note and educational_note.lower() not in explanation.lower():
                enhanced_step["educational_note"] = educational_note
                
            enhanced_steps.append(enhanced_step)
            
        return enhanced_steps
        
    def _extract_essential_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract only the essential steps for a concise solution.
        
        Args:
            steps: All solution steps
            
        Returns:
            Essential steps only
        """
        if not steps:
            return []
            
        # For very short solutions, include all steps
        if len(steps) <= 3:
            return steps
            
        essential_steps = []
        
        # Always include first step
        essential_steps.append(steps[0])
        
        # Find key intermediate steps
        if len(steps) > 2:
            # Identify important operations
            important_operations = ["integrate", "differentiate", "substitute", "factor", "solve"]
            
            # Look for steps with important operations
            key_steps = []
            for i in range(1, len(steps) - 1):
                operation = steps[i].get("operation", "")
                if operation in important_operations:
                    key_steps.append(steps[i])
            
            # If we found key steps, include up to 2 of them
            if key_steps:
                essential_steps.extend(key_steps[:min(len(key_steps), 2)])
            # If no key steps were found, include the middle step
            elif len(steps) >= 4:
                middle_index = len(steps) // 2
                essential_steps.append(steps[middle_index])
                
        # Always include last step
        essential_steps.append(steps[-1])
        
        return essential_steps
        
    def _structure_visualizations(self, 
                                 visualizations: List[Dict[str, Any]], 
                                 complexity_level: str, 
                                 educational: bool = False,
                                 technical: bool = False) -> List[Dict[str, Any]]:
        """
        Structure visualizations based on context and complexity.
        
        Args:
            visualizations: Original visualizations
            complexity_level: Complexity level
            educational: Whether this is for educational format
            technical: Whether this is for technical format
            
        Returns:
            Structured visualizations
        """
        if not visualizations:
            return []
            
        structured_vizs = []
        
        for viz in visualizations:
            structured_viz = viz.copy()
            
            # Enhance visualization based on format
            if educational:
                # Add educational captions
                if "title" in structured_viz and "caption" not in structured_viz:
                    structured_viz["caption"] = f"Figure showing {structured_viz['title'].lower()}. " \
                                               f"Visualizations help develop intuition for mathematical concepts."
                                               
                # Add learning focus for educational context
                structured_viz["learning_focus"] = (
                    "Pay attention to how this visualization represents the mathematical concept, "
                    "and how changes in inputs affect the visual output."
                )
                
            elif technical:
                # Add technical details
                if "parameters" in structured_viz:
                    param_desc = ", ".join(f"{k}={v}" for k, v in structured_viz["parameters"].items())
                    structured_viz["technical_details"] = f"Visualization parameters: {param_desc}"
                    
                # Add formal reference
                if "title" in structured_viz:
                    structured_viz["formal_reference"] = f"Figure: {structured_viz['title']}"
                    
            # Adjust based on complexity
            if complexity_level == "basic":
                # For basic level, add simple explanatory notes
                structured_viz["explanatory_note"] = "This visualization shows the key behavior of the mathematical concept."
                
            elif complexity_level == "advanced":
                # For advanced level, add technical details if not already present
                if "technical_details" not in structured_viz:
                    structured_viz["technical_details"] = (
                        "This visualization represents the precise mathematical relationship "
                        "described in the equations."
                    )
                    
            structured_vizs.append(structured_viz)
            
        return structured_vizs
        
    def _select_best_visualization(self, visualizations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the most informative visualization for concise format.
        
        Args:
            visualizations: All visualizations
            
        Returns:
            Best visualization or None
        """
        if not visualizations:
            return None
            
        # If there's only one visualization, return it
        if len(visualizations) == 1:
            return visualizations[0]
            
        # Define priority order for visualization types
        type_priority = {
            "function_2d": 1,
            "function_with_derivative": 2,
            "function_with_integral": 2,
            "function_3d": 3,
            "vector_field": 4,
            "scatter": 5,
            "histogram": 5,
            "table": 6
        }
        
        # Sort visualizations by priority
        sorted_viz = sorted(
            visualizations, 
            key=lambda v: type_priority.get(v.get("visualization_type", ""), 99)
        )
        
        return sorted_viz[0]
        
    def _format_citations(self, citations: List[str]) -> str:
        """
        Format citations for inclusion in response.
        
        Args:
            citations: Citation information
            
        Returns:
            Formatted citation text
        """
        if not citations:
            return ""
            
        citation_text = "\n\n### References\n\n"
        
        for i, citation in enumerate(citations):
            citation_text += f"{i+1}. {citation}\n"
            
        return citation_text
        
    def _format_minimal_citations(self, citations: List[str]) -> str:
        """
        Format minimal citations for concise responses.
        
        Args:
            citations: Citation information
            
        Returns:
            Minimal citation text
        """
        if not citations:
            return ""
            
        citation_text = "\n\nSources: "
        citation_text += "; ".join(citations)
            
        return citation_text
        
    def _format_formal_citations(self, citations: List[str]) -> str:
        """
        Format formal academic citations.
        
        Args:
            citations: Citation information
            
        Returns:
            Formal citation text
        """
        if not citations:
            return ""
            
        citation_text = "\n\n## References\n\n"
        
        for i, citation in enumerate(citations):
            citation_text += f"[{i+1}] {citation}\n\n"
            
        return citation_text
        
    def _create_conceptual_breakdown(self, 
                                    explanation: str, 
                                    latex_expressions: Union[List[str], str],
                                    steps: List[Dict[str, Any]], 
                                    complexity_level: str) -> str:
        """
        Create a conceptual breakdown section for educational format.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            complexity_level: Complexity level
            
        Returns:
            Conceptual breakdown text
        """
        # Extract key mathematical concepts based on content
        key_concepts = self._extract_key_concepts(explanation, latex_expressions, steps)
        
        if not key_concepts:
            return ""
            
        breakdown_text = "\n\n## Conceptual Breakdown\n\n"
        breakdown_text += "Understanding the following key concepts will help master this topic:\n\n"
        
        for concept, explanation in key_concepts.items():
            breakdown_text += f"**{concept}**: {explanation}\n\n"
            
        if complexity_level == "advanced":
            breakdown_text += (
                "These concepts build upon each other to form a comprehensive understanding "
                "of the mathematical principles involved. Mastering each component will enable "
                "you to tackle more complex problems in this domain."
            )
            
        return breakdown_text
        
    def _extract_key_concepts(self, 
                             explanation: str, 
                             latex_expressions: Union[List[str], str],
                             steps: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract key mathematical concepts from the content.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            
        Returns:
            Dictionary of concept names and explanations
        """
        # This is a simplified implementation
        # In a real system, this would use NLP and domain knowledge
        key_concepts = {}
        
        # Common mathematics concept definitions
        concept_definitions = {
            "derivative": "The rate of change of a function with respect to a variable.",
            "integral": "The area under a curve, or the inverse operation of differentiation.",
            "function": "A relation that associates each input to exactly one output.",
            "limit": "The value a function approaches as the input approaches a specific value.",
            "vector": "A mathematical object with magnitude and direction.",
            "matrix": "A rectangular array of numbers arranged in rows and columns.",
            "probability": "A measure of the likelihood of an event occurring.",
            "equation": "A statement that asserts the equality of two expressions.",
            "variable": "A symbol that represents a number or quantity in a mathematical expression.",
            "polynomial": "An expression consisting of variables and coefficients using addition, subtraction, and multiplication.",
            "logarithm": "The inverse operation to exponentiation.",
            "series": "The sum of the terms of a sequence.",
            "sequence": "An ordered list of numbers or other elements.",
            # Add more concepts as needed
        }
        
        # Look for concept indicators in explanation
        for concept, definition in concept_definitions.items():
            if re.search(r'\b' + concept + r'\b', explanation, re.IGNORECASE):
                key_concepts[concept.capitalize()] = definition
                
        # Look for operation types in steps
        if steps:
            operation_concepts = {
                "differentiate": "Derivative",
                "integrate": "Integral",
                "solve": "Equation Solving",
                "factor": "Factorization",
                "substitute": "Substitution",
                "simplify": "Simplification"
            }
            
            for step in steps:
                operation = step.get("operation", "")
                if operation in operation_concepts and operation_concepts[operation] not in key_concepts:
                    concept = operation_concepts[operation]
                    key_concepts[concept] = concept_definitions.get(
                        concept.lower(), 
                        f"The process of performing {operation} on a mathematical expression."
                    )
                    
        # Limit to at most 5 key concepts to avoid overwhelming
        if len(key_concepts) > 5:
            return dict(list(key_concepts.items())[:5])
            
        return key_concepts
        
    def _create_theoretical_context(self, 
                                   explanation: str, 
                                   latex_expressions: Union[List[str], str],
                                   steps: List[Dict[str, Any]], 
                                   complexity_level: str) -> str:
        """
        Create a theoretical context section for technical format.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            complexity_level: Complexity level
            
        Returns:
            Theoretical context text
        """
        # Extract mathematical domain and theorems
        domain, theorems = self._extract_domain_and_theorems(explanation, latex_expressions, steps)
        
        if not domain and not theorems:
            return ""
            
        context_text = "\n\n## Theoretical Framework\n\n"
        
        if domain:
            context_text += f"This problem is situated within the domain of **{domain}**. "
            
            # Add domain-specific context
            domain_context = {
                "Calculus": "The fundamental theorem of calculus establishes the relationship between differentiation and integration.",
                "Linear Algebra": "The core principles of linear algebra revolve around vector spaces and linear transformations.",
                "Probability": "Probability theory quantifies uncertainty using mathematical models.",
                "Statistics": "Statistical methods analyze and interpret data to extract meaningful insights.",
                "Number Theory": "Number theory explores the properties and relationships of integers.",
                "Differential Equations": "Differential equations describe relationships involving rates of change.",
                # Add more domains as needed
            }
            
            if domain in domain_context:
                context_text += domain_context[domain] + "\n\n"
            else:
                context_text += "\n\n"
                
        if theorems:
            context_text += "The solution applies the following theoretical principles:\n\n"
            for theorem, description in theorems.items():
                context_text += f"**{theorem}**: {description}\n\n"
                
        return context_text
        
    def _extract_domain_and_theorems(self, 
                                    explanation: str, 
                                    latex_expressions: Union[List[str], str],
                                    steps: List[Dict[str, Any]]) -> tuple:
        """
        Extract mathematical domain and theorems from the content.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            
        Returns:
            Tuple of (domain, theorems_dict)
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP and domain knowledge
        
        # Try to identify the mathematical domain
        domain_indicators = {
            "Calculus": ["derivative", "integral", "differentiate", "integrate", "limit"],
            "Linear Algebra": ["matrix", "vector", "linear", "transformation", "eigenvalue"],
            "Probability": ["probability", "random", "distribution", "expectation", "variance"],
            "Statistics": ["statistics", "sample", "mean", "median", "standard deviation"],
            "Number Theory": ["prime", "divisor", "modulo", "congruence", "integer"],
            "Differential Equations": ["differential equation", "ODE", "PDE", "initial value"]
        }
        
        domain = None
        max_indicators = 0
        
        for d, indicators in domain_indicators.items():
            count = sum(1 for ind in indicators if re.search(r'\b' + ind + r'\b', explanation, re.IGNORECASE))
            if count > max_indicators:
                max_indicators = count
                domain = d
                
        # Identify theorems and principles
        theorems = {}
        
        # Common theorems by domain
        domain_theorems = {
            "Calculus": {
                "Fundamental Theorem of Calculus": "Establishes the relationship between differentiation and integration.",
                "Mean Value Theorem": "For a continuous function on a closed interval, there exists a point where the derivative equals the average rate of change over the interval.",
                "Chain Rule": "The derivative of a composite function equals the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function."
            },
            "Linear Algebra": {
                "Rank-Nullity Theorem": "The dimension of the domain equals the rank plus the nullity of a linear transformation.",
                "Invertible Matrix Theorem": "A square matrix is invertible if and only if its determinant is non-zero.",
                "Spectral Theorem": "A symmetric matrix can be diagonalized by an orthogonal matrix."
            },
            # Add more domains and theorems as needed
        }
        
        # Look for theorem mentions in the explanation
        if domain in domain_theorems:
            for theorem, description in domain_theorems[domain].items():
                if theorem.lower() in explanation.lower():
                    theorems[theorem] = description
                    
        # Also look for general theorem mentions
        theorem_keywords = ["theorem", "lemma", "corollary", "principle", "rule"]
        for keyword in theorem_keywords:
            matches = re.finditer(r'\b' + keyword + r'(s)? of ([^.,]+)', explanation, re.IGNORECASE)
            for match in matches:
                theorem_name = match.group(2).strip()
                # Add generic description if theorem found but not in our database
                theorems[f"{theorem_name.capitalize()} {keyword.capitalize()}"] = f"A mathematical {keyword} applicable to this problem."
                
        return domain, theorems
        
    def _generate_learning_objectives(self, 
                                     explanation: str, 
                                     latex_expressions: Union[List[str], str],
                                     steps: List[Dict[str, Any]], 
                                     complexity_level: str) -> List[str]:
        """
        Generate learning objectives for educational format.
        
        Args:
            explanation: Natural language explanation
            latex_expressions: LaTeX expressions
            steps: Solution steps
            complexity_level: Complexity level
            
        Returns:
            List of learning objectives
        """
        # Extract key concepts
        concepts = self._extract_key_concepts(explanation, latex_expressions, steps)
        
        # Extract operations from steps
        operations = set()
        if steps:
            for step in steps:
                operation = step.get("operation", "")
                if operation:
                    operations.add(operation)
                    
        # Generate learning objectives based on complexity
        objectives = []
        
        if complexity_level == "basic":
            objectives.append("Understand the basic concepts involved in this problem.")
            if concepts:
                for concept in list(concepts.keys())[:2]:  # Limit to top 2 concepts for basic level
                    objectives.append(f"Recognize and describe what {concept.lower()} means.")
            if operations:
                for operation in list(operations)[:2]:  # Limit to top 2 operations
                    objectives.append(f"Practice performing {operation} operations in simple contexts.")
                    
        elif complexity_level == "intermediate":
            objectives.append("Apply the mathematical concepts in this problem to solve similar problems.")
            if concepts:
                for concept in list(concepts.keys())[:3]:  # Top 3 concepts
                    objectives.append(f"Explain how {concept.lower()} is applied in this context.")
            if operations:
                for operation in list(operations)[:3]:  # Top 3 operations
                    objectives.append(f"Demonstrate proficiency in {operation} operations.")
                    
        elif complexity_level == "advanced":
            objectives.append("Analyze complex problems using the mathematical principles demonstrated here.")
            if concepts:
                for concept in concepts.keys():  # All concepts for advanced level
                    objectives.append(f"Evaluate how {concept.lower()} relates to other mathematical concepts.")
            if operations:
                for operation in operations:  # All operations
                    objectives.append(f"Apply {operation} techniques to solve advanced problems.")
            objectives.append("Synthesize multiple mathematical concepts to approach novel problems.")
            
        return objectives

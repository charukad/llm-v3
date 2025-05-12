"""
Prompt template builder for mathematical tasks.

This module provides advanced prompt engineering utilities to construct
effective prompts for different mathematical scenarios.
"""
import logging
from typing import Dict, Any, List, Optional, Union

from core.prompting.system_prompts import get_system_prompt
from core.prompting.few_shot_examples import get_examples
from core.prompting.chain_of_thought import get_cot_template, format_cot_prompt
from core.prompting.domain_classifier import classify_domain

logger = logging.getLogger(__name__)

class MathPromptBuilder:
    """
    Builder for constructing optimized mathematical prompts.
    
    This class provides methods to construct effective prompts for
    different mathematical scenarios, incorporating domain-specific
    system instructions, few-shot examples, and chain-of-thought 
    reasoning templates.
    """
    
    def __init__(self):
        """Initialize the prompt builder."""
        pass
    
    def build_mathematical_prompt(
        self,
        query: str,
        domain: Optional[str] = None,
        use_cot: bool = True,
        use_examples: bool = True,
        num_examples: int = 1,
        include_system_prompt: bool = True,
        custom_system_prompt: Optional[str] = None,
        custom_examples: Optional[List[Dict[str, str]]] = None,
        custom_cot_template: Optional[str] = None,
        step_by_step: bool = True,
    ) -> str:
        """
        Build a complete mathematical prompt.
        
        Args:
            query: The mathematical question
            domain: Mathematical domain (algebra, calculus, etc.)
            use_cot: Whether to use chain-of-thought prompting
            use_examples: Whether to include few-shot examples
            num_examples: Number of examples to include
            include_system_prompt: Whether to include the system prompt
            custom_system_prompt: Optional custom system prompt
            custom_examples: Optional custom examples
            custom_cot_template: Optional custom chain-of-thought template
            step_by_step: Whether to explicitly request step-by-step solutions
            
        Returns:
            Formatted prompt optimized for mathematical reasoning
        """
        # Determine domain if not provided
        if domain is None:
            domain = classify_domain(query)
            logger.info(f"Classified query as '{domain}' domain")
        
        # Build the prompt components
        components = []
        
        # 1. Add system prompt if requested
        if include_system_prompt:
            system_prompt = custom_system_prompt or get_system_prompt(domain)
            components.append(system_prompt)
        
        # 2. Add examples if requested
        if use_examples and num_examples > 0:
            examples = custom_examples or get_examples(domain, num_examples)
            example_text = ""
            for i, example in enumerate(examples, 1):
                example_text += f"\nExample {i}:\n"
                example_text += f"Question: {example['question']}\n"
                example_text += f"Solution: {example['answer']}\n"
            
            if example_text:
                components.append(example_text)
        
        # 3. Add the question with appropriate framing
        if step_by_step:
            question_prefix = "Question: "
            answer_prefix = "Solution (step-by-step): "
        else:
            question_prefix = "Question: "
            answer_prefix = "Solution: "
        
        # Create the user query section
        query_section = f"{question_prefix}{query}"
        components.append(query_section)
        
        # 4. Add chain-of-thought template if requested
        if use_cot:
            cot_template = custom_cot_template or get_cot_template(domain)
            components.append(f"{answer_prefix}{cot_template}")
        else:
            components.append(answer_prefix)
        
        # Combine all components
        prompt = "\n\n".join(components)
        
        logger.debug(f"Generated mathematical prompt with {len(components)} components")
        return prompt
    
    def build_verification_prompt(
        self,
        query: str,
        proposed_solution: str,
        domain: Optional[str] = None,
    ) -> str:
        """
        Build a prompt for verifying a mathematical solution.
        
        Args:
            query: The original mathematical question
            proposed_solution: The solution to verify
            domain: Mathematical domain
            
        Returns:
            Formatted prompt for solution verification
        """
        # Determine domain if not provided
        if domain is None:
            domain = classify_domain(query)
        
        verification_system_prompt = """
You are a careful mathematical verification assistant. Your task is to verify mathematical solutions for correctness.
Follow these steps:
1. Read the original question carefully
2. Examine the proposed solution step by step
3. Check for mathematical errors, including calculation mistakes, algebraic errors, or conceptual misunderstandings
4. Verify that the final answer is correct
5. If there are errors, identify and explain them clearly
6. If the solution is correct, confirm its validity

Be thorough and precise in your verification.
"""
        
        prompt = f"""{verification_system_prompt}

Original Question:
{query}

Proposed Solution:
{proposed_solution}

Verification:
Let me verify this solution step by step.
"""
        
        return prompt
    
    def build_hint_prompt(
        self,
        query: str,
        hint_level: str = "medium",
        domain: Optional[str] = None,
    ) -> str:
        """
        Build a prompt for generating a mathematical hint.
        
        Args:
            query: The mathematical question
            hint_level: Level of hint detail ("light", "medium", "strong")
            domain: Mathematical domain
            
        Returns:
            Formatted prompt for hint generation
        """
        # Determine domain if not provided
        if domain is None:
            domain = classify_domain(query)
        
        hint_system_prompt = f"""
You are a helpful mathematical tutor who provides hints that guide students without solving problems for them.

For this {domain} problem, provide a {hint_level}-level hint:
- Light hint: Just point the student toward the general approach or formula
- Medium hint: Give a more specific suggestion about how to start or what technique to use
- Strong hint: Provide a clear direction including the first step, but still leave work for the student

Remember that a good hint enables learning by helping the student make progress without doing the work for them.
"""
        
        prompt = f"""{hint_system_prompt}

Question: {query}

{hint_level.capitalize()} Hint:
"""
        
        return prompt
    
    def build_explanation_prompt(
        self,
        concept: str,
        depth: str = "intermediate",
        include_examples: bool = True,
        domain: Optional[str] = None,
    ) -> str:
        """
        Build a prompt for explaining a mathematical concept.
        
        Args:
            concept: Mathematical concept to explain
            depth: Depth of explanation ("beginner", "intermediate", "advanced")
            include_examples: Whether to include examples
            domain: Mathematical domain
            
        Returns:
            Formatted prompt for concept explanation
        """
        # Determine domain if not provided
        if domain is None:
            domain = classify_domain(concept)
        
        explanation_system_prompt = f"""
You are a mathematics educator who explains concepts clearly and accurately.

Please explain the {domain} concept of "{concept}" at a {depth} level. Your explanation should:
- Define the concept precisely
- Explain its significance and context within {domain}
- Describe the key properties or characteristics
- Connect it to related mathematical ideas
"""
        
        if include_examples:
            explanation_system_prompt += """
- Include concrete examples that illustrate the concept
- If appropriate, use visual descriptions that help with understanding
"""
        
        prompt = f"""{explanation_system_prompt}

Concept: {concept}

Explanation:
"""
        
        return prompt

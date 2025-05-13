"""
Chain-of-Thought prompting templates for the Mathematical Multimodal LLM system.

This module contains templates that encourage step-by-step reasoning
for different types of mathematical problems.
"""
from typing import Dict, Any, Optional

# Base chain-of-thought template for all mathematical problems
BASE_COT_TEMPLATE = """
To solve this problem, I need to:
1. Understand what is being asked
2. Identify the relevant mathematical concepts
3. Break down the problem into steps
4. Solve each step methodically
5. Verify my answer

Let me work through this step-by-step:
"""

# Domain-specific chain-of-thought templates

ALGEBRA_COT_TEMPLATE = """
To solve this algebra problem, I need to:
1. Identify the type of problem (equation, expression, inequality, system)
2. Choose an appropriate algebraic technique
3. Apply transformations systematically
4. Keep track of the domain and restrictions
5. Verify the solution by substitution or checking

Let me work through this methodically:
"""

CALCULUS_COT_TEMPLATE = """
To solve this calculus problem, I need to:
1. Identify whether it involves limits, derivatives, integrals, or series
2. Recall the relevant definitions and theorems
3. Apply appropriate techniques (e.g., product rule, chain rule, u-substitution)
4. Work through the calculation carefully, tracking constants and variables
5. Check my answer for mathematical consistency

Step-by-step solution:
"""

STATISTICS_COT_TEMPLATE = """
To solve this statistics problem, I need to:
1. Identify the probability or statistical concept involved
2. Determine which distribution or method applies
3. Set up the correct formula or approach
4. Calculate carefully, following probability rules
5. Interpret the result in context

My reasoning process:
"""

LINEAR_ALGEBRA_COT_TEMPLATE = """
To solve this linear algebra problem, I need to:
1. Identify the type of problem (matrix operation, linear system, eigenvalues, etc.)
2. Determine the appropriate method or theorem to apply
3. Set up the calculation with careful attention to dimensions
4. Execute the algebraic operations systematically
5. Interpret the result in terms of vector spaces, transformations, or systems

Step-by-step approach:
"""

GEOMETRY_COT_TEMPLATE = """
To solve this geometry problem, I need to:
1. Draw or visualize the geometric situation
2. Identify key properties, theorems, or formulas that apply
3. Establish relationships between the given and unknown quantities
4. Use algebraic, trigonometric, or coordinate methods as appropriate
5. Verify the solution makes geometric sense

Working through this systematically:
"""

# Dictionary mapping domains to their CoT templates
DOMAIN_COT_TEMPLATES = {
    "algebra": ALGEBRA_COT_TEMPLATE,
    "calculus": CALCULUS_COT_TEMPLATE,
    "statistics": STATISTICS_COT_TEMPLATE,
    "linear_algebra": LINEAR_ALGEBRA_COT_TEMPLATE,
    "geometry": GEOMETRY_COT_TEMPLATE,
    "general": BASE_COT_TEMPLATE
}

def get_cot_template(domain: str = "general") -> str:
    """
    Get the appropriate chain-of-thought template for a given mathematical domain.
    
    Args:
        domain: Mathematical domain
        
    Returns:
        Chain-of-thought template for the specified domain
    """
    return DOMAIN_COT_TEMPLATES.get(domain.lower(), BASE_COT_TEMPLATE)

def generate_cot_prompt(prompt: str, system_prompt: str) -> str:
    """
    Generate a Chain-of-Thought prompt by combining system prompt and user prompt.
    
    Args:
        prompt: User input prompt
        system_prompt: System prompt defining model behavior
        
    Returns:
        Combined prompt with CoT instructions
    """
    return f"{system_prompt}\n\nWhen answering, think step-by-step and show your reasoning clearly.\n\n{prompt}"

def format_cot_prompt(
    question: str,
    domain: str = "general",
    examples: Optional[list] = None
) -> str:
    """
    Format a full chain-of-thought prompt with examples and the user question.
    
    Args:
        question: The user's mathematical question
        domain: Mathematical domain
        examples: Optional list of few-shot examples
        
    Returns:
        Formatted prompt with chain-of-thought template
    """
    template = get_cot_template(domain)
    
    # Start with the question
    prompt = f"Question: {question}\n\n"
    
    # Add examples if provided
    if examples:
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"Solution: {example['answer']}\n\n"
    
    # Add the chain-of-thought template
    prompt += f"Solution: {template}\n"
    
    return prompt

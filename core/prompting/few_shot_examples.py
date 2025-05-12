"""
Few-shot examples for the Mathematical Multimodal LLM System.

This module contains examples of mathematical problems and their solutions,
organized by domain, to be used for few-shot prompting.
"""
from typing import List, Dict, Any

# Algebra examples
ALGEBRA_EXAMPLES = [
    {
        "question": "Solve the quadratic equation: 2x² - 5x - 3 = 0",
        "answer": """I'll solve the quadratic equation 2x² - 5x - 3 = 0 using the quadratic formula.

Step 1: Identify the coefficients.
   a = 2, b = -5, c = -3

Step 2: Apply the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)
   x = (5 ± √((-5)² - 4(2)(-3))) / (2(2))
   x = (5 ± √(25 + 24)) / 4
   x = (5 ± √49) / 4
   x = (5 ± 7) / 4

Step 3: Calculate the two solutions.
   x₁ = (5 + 7) / 4 = 12/4 = 3
   x₂ = (5 - 7) / 4 = -2/4 = -1/2

Step 4: Verify the solutions by substituting back into the original equation.
   For x = 3:
   2(3)² - 5(3) - 3 = 2(9) - 15 - 3 = 18 - 15 - 3 = 0 ✓
   
   For x = -1/2:
   2(-1/2)² - 5(-1/2) - 3 = 2(1/4) + 5/2 - 3 = 1/2 + 5/2 - 3 = 3 - 3 = 0 ✓

Therefore, the solutions to 2x² - 5x - 3 = 0 are x = 3 and x = -1/2."""
    },
    {
        "question": "Simplify the expression: (3x² - 2x + 1) - (x² + 4x - 2)",
        "answer": """I'll simplify the expression (3x² - 2x + 1) - (x² + 4x - 2) by combining like terms.

Step 1: Apply the distributive property to distribute the negative sign.
   (3x² - 2x + 1) - (x² + 4x - 2)
   = 3x² - 2x + 1 - x² - 4x + 2

Step 2: Group like terms.
   = (3x² - x²) + (-2x - 4x) + (1 + 2)
   = 2x² - 6x + 3

Therefore, (3x² - 2x + 1) - (x² + 4x - 2) simplifies to 2x² - 6x + 3."""
    },
    {
        "question": "Solve the system of equations: 3x + 2y = 7 and 2x - y = 4",
        "answer": """I'll solve this system of equations using the elimination method.

Step 1: Multiply the second equation by 2 to align the y terms for elimination.
   3x + 2y = 7       (Equation 1)
   2x - y = 4        (Equation 2)
   
   Multiply Equation 2 by 2:
   4x - 2y = 8       (Equation 2')

Step 2: Add Equation 1 and Equation 2' to eliminate y.
   3x + 2y = 7
   4x - 2y = 8
   ---------------
   7x + 0 = 15
   
   Therefore, x = 15/7 = 2.14... = 15/7

Step 3: Substitute x = 15/7 back into Equation 2 to find y.
   2(15/7) - y = 4
   30/7 - y = 4
   -y = 4 - 30/7
   -y = 28/7 - 30/7
   -y = -2/7
   y = 2/7

Step 4: Verify the solution by substituting back into both equations.
   Equation 1: 3(15/7) + 2(2/7) = 45/7 + 4/7 = 49/7 = 7 ✓
   Equation 2: 2(15/7) - (2/7) = 30/7 - 2/7 = 28/7 = 4 ✓

Therefore, the solution to the system is x = 15/7 and y = 2/7."""
    }
]

# Calculus examples
CALCULUS_EXAMPLES = [
    {
        "question": "Find the derivative of f(x) = x³ - 4x² + 5x - 2",
        "answer": """I'll find the derivative of f(x) = x³ - 4x² + 5x - 2 using the power rule and linearity of differentiation.

The power rule states that for any term x^n, the derivative is n·x^(n-1).

Step 1: Apply the power rule to each term.
   f(x) = x³ - 4x² + 5x - 2
   
   For x³: The derivative is 3x²
   For -4x²: The derivative is -4 · 2x = -8x
   For 5x: The derivative is 5 · 1 = 5
   For -2: The derivative is 0 (constants have zero derivative)

Step 2: Combine the terms.
   f'(x) = 3x² - 8x + 5

Therefore, the derivative of f(x) = x³ - 4x² + 5x - 2 is f'(x) = 3x² - 8x + 5."""
    },
    {
        "question": "Evaluate the integral: ∫(2x + 3)dx",
        "answer": """I'll evaluate the indefinite integral ∫(2x + 3)dx using the basic rules of integration.

Step 1: Apply the linearity of integration to separate the terms.
   ∫(2x + 3)dx = ∫2x dx + ∫3 dx

Step 2: Apply the power rule for integration to each term.
   The power rule for integration states that ∫x^n dx = (x^(n+1))/(n+1) + C for n ≠ -1.
   
   For ∫2x dx: Using the power rule with n = 1, we get 2 · (x²/2) = x²
   For ∫3 dx: Integrating a constant gives 3x

Step 3: Combine terms and add the constant of integration.
   ∫(2x + 3)dx = x² + 3x + C

Therefore, the indefinite integral ∫(2x + 3)dx = x² + 3x + C, where C is the constant of integration."""
    },
    {
        "question": "Find the limit: lim(x→2) (x² - 4)/(x - 2)",
        "answer": """I'll find the limit: lim(x→2) (x² - 4)/(x - 2)

Step 1: Try direct substitution first.
   When x = 2, we get (2² - 4)/(2 - 2) = (4 - 4)/0 = 0/0, which is indeterminate.

Step 2: Since we have an indeterminate form 0/0, I'll factor the numerator.
   x² - 4 = (x - 2)(x + 2)
   
   So the expression becomes:
   (x² - 4)/(x - 2) = [(x - 2)(x + 2)]/(x - 2) = x + 2

Step 3: Now we can evaluate the limit by substitution.
   lim(x→2) (x² - 4)/(x - 2) = lim(x→2) (x + 2) = 2 + 2 = 4

Therefore, lim(x→2) (x² - 4)/(x - 2) = 4."""
    }
]

# Statistics examples
STATISTICS_EXAMPLES = [
    {
        "question": "A normal distribution has a mean of 70 and a standard deviation of 5. What is the probability that a random observation exceeds 75?",
        "answer": """I'll find the probability that a random observation from a normal distribution with mean μ = 70 and standard deviation σ = 5 exceeds 75.

Step 1: Standardize the value by converting to a z-score.
   z = (x - μ) / σ
   z = (75 - 70) / 5
   z = 5 / 5
   z = 1

Step 2: Find the probability using the standard normal distribution.
   We want P(X > 75) = P(Z > 1)
   
   For the standard normal distribution, P(Z > 1) = 1 - P(Z ≤ 1)
   P(Z ≤ 1) = 0.8413 (using the standard normal table or calculator)
   
   Therefore, P(Z > 1) = 1 - 0.8413 = 0.1587

Therefore, the probability that a random observation exceeds 75 is 0.1587, or approximately 15.87%."""
    },
    {
        "question": "A coin is flipped 10 times. What is the probability of getting exactly 6 heads?",
        "answer": """I'll find the probability of getting exactly 6 heads in 10 coin flips.

Step 1: Identify that this is a binomial probability problem.
   - We have a fixed number of trials (n = 10)
   - Each trial has two possible outcomes (heads or tails)
   - The trials are independent
   - The probability of success (heads) is constant (p = 0.5)

Step 2: Apply the binomial probability formula:
   P(X = k) = (n choose k) × p^k × (1-p)^(n-k)
   
   Where (n choose k) is the binomial coefficient: n! / [k! × (n-k)!]

Step 3: Calculate P(X = 6) with n = 10, k = 6, and p = 0.5
   P(X = 6) = (10 choose 6) × 0.5^6 × 0.5^4
   
   First, calculate the binomial coefficient:
   (10 choose 6) = 10! / [6! × (10-6)!]
   = 10! / [6! × 4!]
   = (10 × 9 × 8 × 7) / (4 × 3 × 2 × 1)
   = 210
   
   Now calculate the full probability:
   P(X = 6) = 210 × 0.5^6 × 0.5^4
   = 210 × 0.5^10
   = 210 × 0.000976
   = 0.205

Therefore, the probability of getting exactly 6 heads in 10 coin flips is 0.205, or approximately 20.5%."""
    }
]

# Linear Algebra examples
LINEAR_ALGEBRA_EXAMPLES = [
    {
        "question": "Find the eigenvalues and eigenvectors of the matrix A = [[2, 1], [1, 2]]",
        "answer": """I'll find the eigenvalues and eigenvectors of the matrix A = [[2, 1], [1, 2]].

Step 1: Find the eigenvalues by solving the characteristic equation |A - λI| = 0.
   A - λI = [[2-λ, 1], [1, 2-λ]]
   
   |A - λI| = (2-λ)(2-λ) - 1×1
   = (2-λ)² - 1
   = 4 - 4λ + λ² - 1
   = λ² - 4λ + 3
   = (λ - 3)(λ - 1)
   
   Setting this equal to zero:
   (λ - 3)(λ - 1) = 0
   λ = 3 or λ = 1

So the eigenvalues are λ₁ = 3 and λ₂ = 1.

Step 2: Find the eigenvectors for each eigenvalue by solving (A - λI)v = 0.

For λ₁ = 3:
   A - 3I = [[2-3, 1], [1, 2-3]] = [[-1, 1], [1, -1]]
   
   We need to solve:
   -v₁ + v₂ = 0
   v₁ - v₂ = 0
   
   These equations are identical (both give v₁ = v₂), so any vector where v₁ = v₂ is an eigenvector.
   Let's choose v₁ = 1, which gives v₂ = 1.
   
   So, an eigenvector for λ₁ = 3 is v₁ = [1, 1].

For λ₂ = 1:
   A - 1I = [[2-1, 1], [1, 2-1]] = [[1, 1], [1, 1]]
   
   We need to solve:
   v₁ + v₂ = 0
   v₁ + v₂ = 0
   
   This gives v₂ = -v₁. Let's choose v₁ = 1, which gives v₂ = -1.
   
   So, an eigenvector for λ₂ = 1 is v₂ = [1, -1].

Therefore:
- The eigenvalues of A are λ₁ = 3 and λ₂ = 1
- An eigenvector for λ₁ = 3 is [1, 1]
- An eigenvector for λ₂ = 1 is [1, -1]"""
    },
    {
        "question": "Solve the system of linear equations: 2x + y = 5, 3x - 2y = 4",
        "answer": """I'll solve the system of linear equations:
2x + y = 5
3x - 2y = 4

Step 1: Solve for y in terms of x using the first equation.
2x + y = 5
y = 5 - 2x

Step 2: Substitute this expression for y into the second equation.
3x - 2(5 - 2x) = 4
3x - 10 + 4x = 4
7x - 10 = 4
7x = 14
x = 2

Step 3: Substitute x = 2 back into our expression for y.
y = 5 - 2(2)
y = 5 - 4
y = 1

Step 4: Verify the solution by checking both original equations.
2(2) + 1 = 5 ✓
3(2) - 2(1) = 6 - 2 = 4 ✓

Therefore, the solution to the system is x = 2 and y = 1."""
    }
]

# Geometry examples
GEOMETRY_EXAMPLES = [
    {
        "question": "Find the area of a circle with radius 5 units.",
        "answer": """I'll find the area of a circle with radius 5 units.

Step 1: Recall the formula for the area of a circle.
   Area = πr²
   Where r is the radius of the circle.

Step 2: Substitute the radius r = 5 into the formula.
   Area = π(5)²
   Area = π × 25
   Area = 25π

Therefore, the area of a circle with radius 5 units is 25π square units, which is approximately 78.54 square units."""
    },
    {
        "question": "In a right triangle, if one leg is 6 and the hypotenuse is 10, find the length of the other leg.",
        "answer": """I'll find the length of the other leg of a right triangle with one leg = 6 and hypotenuse = 10.

Step 1: Recall the Pythagorean theorem for right triangles.
   a² + b² = c²
   Where a and b are the legs, and c is the hypotenuse.

Step 2: Substitute the known values into the formula.
   Let's denote the unknown leg as b.
   6² + b² = 10²
   36 + b² = 100
   
Step 3: Solve for b.
   b² = 100 - 36
   b² = 64
   b = 8 (since b is a length, we take the positive square root)

Step 4: Verify the solution using the Pythagorean theorem.
   a² + b² = c²
   6² + 8² = 10²
   36 + 64 = 100
   100 = 100 ✓

Therefore, the length of the other leg is 8 units."""
    }
]

# Dictionary mapping domains to their example sets
DOMAIN_EXAMPLES = {
    "algebra": ALGEBRA_EXAMPLES,
    "calculus": CALCULUS_EXAMPLES,
    "statistics": STATISTICS_EXAMPLES,
    "linear_algebra": LINEAR_ALGEBRA_EXAMPLES,
    "geometry": GEOMETRY_EXAMPLES,
}

def get_examples(domain: str, num_examples: int = 2) -> List[Dict[str, str]]:
    """
    Get a specified number of few-shot examples for a given domain.
    
    Args:
        domain: Mathematical domain
        num_examples: Number of examples to return
        
    Returns:
        List of example dictionaries, each containing a question and answer
    """
    examples = DOMAIN_EXAMPLES.get(domain.lower(), [])
    # Return up to num_examples, but don't fail if fewer are available
    return examples[:min(num_examples, len(examples))]

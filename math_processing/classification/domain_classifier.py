"""
Mathematical Domain Classifier - identifies the mathematical domain of a query.

This module provides functionality to classify mathematical queries into 
specific domains such as algebra, calculus, linear algebra, etc., to enable
specialized processing and context-aware responses.
"""

import re
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
import os


class MathDomainClassifier:
    """Classifier for mathematical domains."""
    
    def __init__(self, custom_keywords_path: Optional[str] = None):
        """
        Initialize the domain classifier.
        
        Args:
            custom_keywords_path: Path to custom keywords file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load domain keywords
        self.domain_keywords = self._load_domain_keywords()
        
        # Load custom keywords if provided
        if custom_keywords_path and os.path.exists(custom_keywords_path):
            self._load_custom_keywords(custom_keywords_path)
        
        # Define domain descriptions
        self.domain_descriptions = {
            "algebra": "Algebra deals with symbols and the rules for manipulating these symbols, including equations, polynomials, and factoring.",
            "calculus": "Calculus is the mathematical study of continuous change, involving derivatives, integrals, limits, and infinite series.",
            "linear_algebra": "Linear algebra concerns vector spaces and linear mappings between them, including systems of linear equations, matrices, and determinants.",
            "statistics": "Statistics involves the collection, analysis, interpretation, and presentation of data, including probability, distributions, and hypothesis testing.",
            "geometry": "Geometry studies the properties and relationships of points, lines, surfaces, and solids, including angles, distances, and shapes.",
            "number_theory": "Number theory deals with the properties and relationships of numbers, particularly integers, including prime numbers and modular arithmetic.",
            "trigonometry": "Trigonometry studies relationships between side lengths and angles of triangles, including trigonometric functions and identities.",
            "discrete_math": "Discrete mathematics deals with mathematical structures that are fundamentally discrete rather than continuous, including combinatorics and graph theory."
        }
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """
        Load domain keywords for classification.
        
        Returns:
            Dictionary mapping domains to lists of keywords
        """
        # This would typically load from a file in a real implementation
        # Here we define them inline for simplicity
        
        keywords = {
            "algebra": [
                "equation", "solve", "polynomial", "factor", "expand", "simplify", 
                "quadratic", "linear equation", "system of equations", "factorize",
                "binomial", "trinomial", "roots", "coefficients", "discriminant",
                "completing the square", "rational expression", "algebraic"
            ],
            
            "calculus": [
                "derivative", "differentiate", "integrate", "integration", "limit",
                "d/dx", "dy/dx", "antiderivative", "chain rule", "product rule",
                "quotient rule", "power rule", "fundamental theorem", "continuous",
                "differential", "rate of change", "tangent", "maximum", "minimum",
                "inflection point", "concave", "convex", "series", "sequence",
                "converge", "taylor", "maclaurin", "partial derivative", "gradient"
            ],
            
            "linear_algebra": [
                "matrix", "vector", "determinant", "eigenvalue", "eigenvector",
                "linear system", "linear transformation", "basis", "span", "invertible",
                "singular", "transpose", "orthogonal", "orthonormal", "projection",
                "subspace", "null space", "row space", "column space", "rank",
                "dimension", "identity matrix", "diagonal", "lower triangular",
                "upper triangular", "system of linear equations", "augmented matrix",
                "Gaussian elimination", "row echelon form"
            ],
            
            "statistics": [
                "probability", "distribution", "mean", "median", "mode", "variance",
                "standard deviation", "hypothesis test", "p-value", "confidence interval",
                "significance level", "null hypothesis", "alternative hypothesis",
                "correlation", "regression", "sample", "population", "t-test",
                "chi-square", "normal distribution", "binomial", "poisson",
                "expected value", "random variable", "bayes", "conditional probability",
                "joint probability", "independent", "dependent", "frequency"
            ],
            
            "geometry": [
                "angle", "triangle", "circle", "polygon", "perimeter", "area",
                "volume", "parallel", "perpendicular", "similar", "congruent",
                "coordinate", "distance", "point", "line", "plane", "cone",
                "sphere", "cylinder", "rectangle", "square", "regular", "circumference",
                "diameter", "radius", "hypotenuse", "pythagorean", "coordinate geometry",
                "cartesian", "transformation", "rotation", "reflection", "translation",
                "dilation", "euclidean"
            ],
            
            "number_theory": [
                "prime", "composite", "factor", "multiple", "divisor", "divisible",
                "modulo", "congruence", "gcd", "lcm", "greatest common divisor",
                "least common multiple", "coprime", "relatively prime", "fermat",
                "euler", "parity", "even", "odd", "remainder", "divisibility",
                "fundamental theorem of arithmetic", "prime factorization",
                "diophantine", "integer", "natural number"
            ],
            
            "trigonometry": [
                "sine", "cosine", "tangent", "sin", "cos", "tan", "sec", "csc", "cot",
                "secant", "cosecant", "cotangent", "radian", "degree", "angle",
                "triangle", "right triangle", "pythagorean", "identity", "law of sines",
                "law of cosines", "periodic", "amplitude", "frequency", "phase",
                "unit circle", "trigonometric", "inverse trigonometric", "arcsin",
                "arccos", "arctan", "periodic function"
            ],
            
            "discrete_math": [
                "combinatorics", "permutation", "combination", "factorial", "choose",
                "graph", "vertex", "edge", "path", "cycle", "tree", "forest",
                "connected", "disconnected", "adjacency", "isomorphic", "bijection",
                "recurrence", "recursive", "sequence", "set", "subset", "union",
                "intersection", "complement", "difference", "boolean", "logic",
                "proposition", "predicate", "quantifier", "induction", "pigeonhole",
                "principle", "modular arithmetic"
            ]
        }
        
        return keywords
    
    def _load_custom_keywords(self, file_path: str) -> None:
        """
        Load custom keywords from a JSON file.
        
        Args:
            file_path: Path to the JSON file with custom keywords
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_keywords = json.load(f)
            
            # Merge custom keywords with default keywords
            for domain, keywords in custom_keywords.items():
                if domain in self.domain_keywords:
                    # Add new keywords to existing domain
                    self.domain_keywords[domain].extend(keywords)
                    # Remove duplicates
                    self.domain_keywords[domain] = list(set(self.domain_keywords[domain]))
                else:
                    # Add new domain
                    self.domain_keywords[domain] = keywords
            
            self.logger.info(f"Loaded custom keywords from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load custom keywords: {str(e)}")
    
    def classify_query(self, 
                     query: str, 
                     include_details: bool = False) -> Dict[str, Any]:
        """
        Classify a mathematical query into domains.
        
        Args:
            query: Query text to classify
            include_details: Whether to include detailed classification information
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Clean and normalize the query
            normalized_query = self._preprocess_query(query)
            
            # Count keyword matches for each domain
            matches = {}
            for domain, keywords in self.domain_keywords.items():
                domain_matches = []
                for keyword in keywords:
                    # Look for word boundary matches to avoid partial matches
                    # e.g. "integrate" should match "integrate" but not "integrated circuit"
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches_found = re.findall(pattern, normalized_query, re.IGNORECASE)
                    if matches_found:
                        domain_matches.extend(matches_found)
                
                # Store the number of unique matches
                matches[domain] = len(set(domain_matches))
            
            # Calculate confidence scores
            total_matches = sum(matches.values())
            confidence_scores = {}
            
            if total_matches > 0:
                for domain, count in matches.items():
                    confidence_scores[domain] = count / total_matches
            
            # Identify primary and secondary domains based on confidence scores
            sorted_domains = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            
            primary_domain = sorted_domains[0][0] if sorted_domains else "unknown"
            primary_confidence = sorted_domains[0][1] if sorted_domains else 0
            
            secondary_domains = [domain for domain, score in sorted_domains[1:3] if score > 0]
            
            # Handle the case where no matches were found
            if primary_confidence == 0:
                primary_domain = "algebra"  # Default domain if no matches
                primary_confidence = 0.5  # Moderate confidence
            
            # Prepare the response
            result = {
                "primary_domain": primary_domain,
                "primary_confidence": primary_confidence,
                "secondary_domains": secondary_domains,
                "domain_description": self.domain_descriptions.get(primary_domain, "")
            }
            
            # Include detailed information if requested
            if include_details:
                result["confidence_scores"] = confidence_scores
                result["domain_matches"] = matches
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying query: {str(e)}")
            return {
                "primary_domain": "unknown",
                "primary_confidence": 0,
                "secondary_domains": [],
                "error": str(e)
            }
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess a query for classification.
        
        Args:
            query: Original query text
            
        Returns:
            Preprocessed query text
        """
        # Convert to lowercase
        text = query.lower()
        
        # Replace common mathematical symbols with their names to improve matching
        replacements = {
            "∫": " integrate ",
            "∂": " partial derivative ",
            "∑": " sum ",
            "∏": " product ",
            "√": " square root ",
            "∞": " infinity ",
            "≠": " not equal ",
            "≤": " less than or equal ",
            "≥": " greater than or equal ",
            "→": " approaches ",
            "∈": " element of ",
            "∪": " union ",
            "∩": " intersection ",
            "⊂": " subset ",
            "⊃": " superset ",
            "d/dx": " derivative ",
            "dy/dx": " derivative ",
            "lim": " limit ",
            "sin": " sine ",
            "cos": " cosine ",
            "tan": " tangent "
        }
        
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, replacement)
        
        # Remove special characters that aren't relevant for classification
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_domain_examples(self, domain: str, count: int = 3) -> List[str]:
        """
        Get example queries for a specific domain.
        
        Args:
            domain: Mathematical domain
            count: Number of examples to return
            
        Returns:
            List of example queries
        """
        examples = {
            "algebra": [
                "Solve the equation 2x + 5 = 13",
                "Factor the polynomial x^2 + 5x + 6",
                "Simplify the expression (x^2 - 4)/(x - 2)",
                "Solve the system of equations: 2x + y = 10, x - y = 4",
                "Find the roots of the quadratic equation 3x^2 - 6x + 2 = 0"
            ],
            
            "calculus": [
                "Find the derivative of f(x) = x^3 sin(x)",
                "Calculate the integral of 2x + e^x with respect to x",
                "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1",
                "Find the local maxima and minima of f(x) = x^3 - 3x^2 + 2",
                "Calculate the Taylor series expansion of e^x around x = 0"
            ],
            
            "linear_algebra": [
                "Find the determinant of the matrix [[1, 2], [3, 4]]",
                "Calculate the eigenvalues of the matrix [[4, 2], [1, 3]]",
                "Solve the system of linear equations: 2x + 3y = 7, 4x - y = 5",
                "Find the inverse of the matrix [[2, 1], [1, 3]]",
                "Calculate the dot product of vectors [1, 2, 3] and [4, 5, 6]"
            ],
            
            "statistics": [
                "Calculate the mean and standard deviation of [12, 15, 18, 22, 30]",
                "Find the probability of rolling a sum of 7 with two dice",
                "Perform a hypothesis test with significance level 0.05",
                "Calculate the correlation coefficient between X and Y",
                "Find the expected value of a discrete random variable"
            ],
            
            "geometry": [
                "Find the area of a triangle with sides 3, 4, and 5",
                "Calculate the distance between points (1, 2) and (4, 6)",
                "Find the equation of a circle with center (2, 3) and radius 4",
                "Calculate the volume of a cone with radius 3 and height 7",
                "Find the angle between two vectors [1, 0] and [0, 1]"
            ],
            
            "number_theory": [
                "Find all prime numbers less than 50",
                "Calculate the greatest common divisor of 48 and 36",
                "Determine if 127 is a prime number",
                "Find the remainder when 17^43 is divided by 7",
                "Find all solutions to the congruence 3x ≡ 4 (mod 7)"
            ],
            
            "trigonometry": [
                "Calculate sin(30°) and cos(45°)",
                "Verify the identity sin^2(x) + cos^2(x) = 1",
                "Solve the equation 2sin(x) + 1 = 0 for 0 ≤ x < 2π",
                "Find the values of all six trigonometric functions at π/4",
                "Calculate the area of a triangle using the law of sines"
            ],
            
            "discrete_math": [
                "Calculate the number of ways to arrange 5 different books on a shelf",
                "Find the number of subsets of a set with 6 elements",
                "Determine if the graph with edges {(1,2), (2,3), (3,4), (4,1)} is connected",
                "Solve the recurrence relation a_n = a_{n-1} + a_{n-2} with a_0 = 0 and a_1 = 1",
                "Find the truth table for the logical expression (p ∧ q) → (p ∨ ¬q)"
            ]
        }
        
        if domain in examples:
            return examples[domain][:count]
        else:
            return [f"No examples available for domain: {domain}"]
    
    def get_all_domains(self) -> List[str]:
        """
        Get a list of all supported domains.
        
        Returns:
            List of domain names
        """
        return list(self.domain_keywords.keys())
    
    def get_domain_description(self, domain: str) -> str:
        """
        Get a description of a specific domain.
        
        Args:
            domain: Mathematical domain
            
        Returns:
            Description of the domain
        """
        return self.domain_descriptions.get(domain, f"No description available for domain: {domain}")

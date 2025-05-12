"""
Domain classification for mathematical queries.

This module handles classifying mathematical questions into appropriate domains
to apply domain-specific prompting and reasoning.
"""
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Keywords and phrases associated with each mathematical domain
DOMAIN_KEYWORDS = {
    "algebra": [
        "equation", "solve for", "simplify", "factor", "polynomial", "linear", 
        "quadratic", "cubic", "inequality", "system of equations", "expression",
        "variable", "coefficient", "term", "expand", "binomial", "trinomial",
        "exponent", "logarithm", "root", "radical", "substitution"
    ],
    "calculus": [
        "derivative", "differentiate", "integrate", "integration", "antiderivative",
        "limit", "approaches", "converge", "diverge", "series", "sequence",
        "continuous", "discontinuous", "maximum", "minimum", "critical point",
        "inflection point", "differential", "rate of change", "velocity",
        "acceleration", "tangent", "normal", "partial", "gradient", "divergence"
    ],
    "statistics": [
        "probability", "random", "distribution", "normal", "binomial", "poisson",
        "mean", "median", "mode", "expected value", "variance", "standard deviation",
        "hypothesis", "test", "p-value", "confidence interval", "significant",
        "sample", "population", "correlation", "regression", "percentile", "quartile"
    ],
    "linear_algebra": [
        "matrix", "vector", "determinant", "eigenvalue", "eigenvector",
        "transformation", "linear", "basis", "dimension", "orthogonal",
        "orthonormal", "projection", "span", "linear combination", "row",
        "column", "scalar", "dot product", "cross product", "transpose",
        "inverse", "diagonalization", "trace", "rank", "nullity", "kernel"
    ],
    "geometry": [
        "triangle", "circle", "square", "rectangle", "polygon", "angle",
        "parallel", "perpendicular", "distance", "coordinate", "line",
        "point", "plane", "volume", "area", "perimeter", "circumference",
        "congruent", "similar", "symmetry", "transformation", "rotation",
        "reflection", "translation", "dilation", "vector", "trigonometry"
    ]
}

def classify_domain(query: str) -> str:
    """
    Classify a mathematical query into a specific domain.
    
    Args:
        query: The mathematical question to classify
        
    Returns:
        The most likely mathematical domain
    """
    query = query.lower()
    
    # Count keyword occurrences for each domain
    domain_scores: Dict[str, int] = {domain: 0 for domain in DOMAIN_KEYWORDS}
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in query:
                domain_scores[domain] += 1
    
    # Find the domain with the highest score
    max_score = 0
    best_domain = "general"
    
    for domain, score in domain_scores.items():
        if score > max_score:
            max_score = score
            best_domain = domain
    
    logger.info(f"Classified query as '{best_domain}' domain with score {max_score}")
    logger.debug(f"Domain scores: {domain_scores}")
    
    # If we don't have a clear signal, fall back to "general"
    if max_score < 2:
        logger.info(f"Score too low, falling back to 'general' domain")
        return "general"
    
    return best_domain

def extract_mathematical_entities(query: str) -> List[str]:
    """
    Extract mathematical entities from a query for better processing.
    
    Args:
        query: The mathematical question to analyze
        
    Returns:
        List of identified mathematical entities
    """
    # This is a simplified implementation
    # In a real system, you would use more sophisticated NLP techniques
    
    entities = []
    
    # Look for expressions wrapped in various delimiters
    delimiters = [
        ("$", "$"),
        ("\\(", "\\)"),
        ("\\[", "\\]"),
        ("{", "}"),
    ]
    
    for start_delim, end_delim in delimiters:
        start_idx = 0
        while True:
            start_idx = query.find(start_delim, start_idx)
            if start_idx == -1:
                break
            
            end_idx = query.find(end_delim, start_idx + len(start_delim))
            if end_idx == -1:
                break
            
            entity = query[start_idx + len(start_delim):end_idx]
            entities.append(entity)
            start_idx = end_idx + len(end_delim)
    
    # Look for common mathematical operators and their surroundings
    operators = ["+", "-", "*", "/", "=", "<", ">", "^", "\\frac", "\\sqrt"]
    words = query.split()
    
    for i, word in enumerate(words):
        for op in operators:
            if op in word:
                # Try to capture a window around the operator
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                entity = " ".join(words[start:end])
                if entity not in entities:
                    entities.append(entity)
    
    return entities

def get_detailed_classification(query: str) -> Dict[str, Any]:
    """
    Get a detailed classification of a mathematical query.
    
    Args:
        query: The mathematical question to classify
        
    Returns:
        Dictionary with domain, confidence, and entities
    """
    # Classify the primary domain
    primary_domain = classify_domain(query)
    
    # Get domain scores for confidence
    domain_scores: Dict[str, int] = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in query.lower():
                domain_scores[domain] += 1
    
    # Calculate confidence as the ratio of the highest score to the total
    total_score = sum(domain_scores.values())
    confidence = domain_scores[primary_domain] / max(total_score, 1)
    
    # Extract entities
    entities = extract_mathematical_entities(query)
    
    return {
        "domain": primary_domain,
        "confidence": confidence,
        "domain_scores": domain_scores,
        "entities": entities
    }

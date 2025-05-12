#!/usr/bin/env python3
"""
Test script for pattern extraction logic.
"""
import re

def extract_expression(text):
    """Extract mathematical expressions from text."""
    # Define patterns
    expression_patterns = [
        r'sin\s*\(\s*x\s*\)',  # sin(x)
        r'cos\s*\(\s*x\s*\)',  # cos(x)
        r'tan\s*\(\s*x\s*\)',  # tan(x)
        r'x\s*\^\s*\d+',       # x^2, x^3, etc.
        r'e\s*\^\s*x',         # e^x
        r'log\s*\(\s*x\s*\)',  # log(x)
        r'ln\s*\(\s*x\s*\)',   # ln(x)
        r'f\s*\(\s*x\s*\)\s*=\s*([^\n]+)',  # f(x) = ...
        r'y\s*=\s*([^\n]+)'    # y = ...
    ]
    
    # Try each pattern
    for pattern in expression_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if '=' in pattern:
                return match.group(1).strip()
            else:
                return match.group(0).strip()
    
    # Fallback heuristics
    if "sin" in text.lower():
        return "sin(x)"
    elif "cos" in text.lower():
        return "cos(x)"
    elif "plot" in text.lower():
        return "x^2"  # Default fallback
    
    return None

# Test cases
test_cases = [
    "Can you plot sin(x) for me?",
    "Plot sinx from -10 to 10",
    "I want to see a graph of sin ( x )",
    "Could you draw cos(x)?",
    "Visualize x^2 - 2x + 3",
    "Show me e^x",
    "Plot f(x) = 2x + 3",
    "y = x^3 - 4x graph",
]

for test in test_cases:
    expression = extract_expression(test)
    print(f"Text: {test}")
    print(f"Extracted: {expression}")
    print("-" * 50)

# Test our specific case in isolation
text = "Can you plot sin(x) for me?"
for pattern in [r'sin\s*\(\s*x\s*\)']:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        print(f"Direct match for '{pattern}' in '{text}':")
        print(f"Match: {match.group(0)}")
    else:
        print(f"No match for '{pattern}' in '{text}'") 
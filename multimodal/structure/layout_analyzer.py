"""
Layout analyzer for mathematical expressions.

This module analyzes the spatial structure of detected symbols to
understand mathematical expressions.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import math

logger = logging.getLogger(__name__)

def analyze_layout(symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the 2D layout of mathematical symbols.
    
    Args:
        symbols: List of detected symbols with positions
        
    Returns:
        Dictionary containing the expression structure
    """
    try:
        # Sort symbols by x-coordinate to establish reading order
        sorted_symbols = sorted(symbols, key=lambda s: s["position"][0])
        
        # Group symbols into lines
        lines = group_into_lines(sorted_symbols)
        
        # Analyze structures within each line
        structures = []
        for line in lines:
            structure = analyze_line_structure(line)
            structures.append(structure)
        
        # Combine line structures
        if len(structures) == 1:
            result = structures[0]
        else:
            result = {
                "type": "multi_line",
                "lines": structures
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing layout: {str(e)}")
        return {
            "type": "error",
            "error": str(e)
        }

def group_into_lines(symbols: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group symbols into lines based on vertical position.
    
    Args:
        symbols: List of detected symbols
        
    Returns:
        List of symbol lines
    """
    if not symbols:
        return []
    
    # Calculate average symbol height
    avg_height = sum(s["position"][3] for s in symbols) / len(symbols)
    
    # Group symbols by vertical position (y-coordinate)
    lines = []
    current_line = [symbols[0]]
    current_line_y = symbols[0]["position"][1]
    
    for symbol in symbols[1:]:
        y_pos = symbol["position"][1]
        
        # If y position is within tolerance of current line, add to line
        if abs(y_pos - current_line_y) < avg_height * 0.5:
            current_line.append(symbol)
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [symbol]
            current_line_y = y_pos
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    # Sort each line by x-coordinate
    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda s: s["position"][0])
    
    return lines

def analyze_line_structure(line: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the structure of a line of symbols.
    
    Args:
        line: List of symbols in a line
        
    Returns:
        Dictionary representing the structure
    """
    if not line:
        return {"type": "empty"}
    
    # Check for explicit relations in symbols (from advanced detection)
    elements = []
    i = 0
    
    while i < len(line):
        symbol = line[i]
        
        # Check symbol relations
        if "relation" in symbol:
            relation = symbol["relation"]
            
            if relation == "superscript" and "related_to" in symbol:
                # Find the base symbol
                base_idx = symbol["related_to"]
                base_symbol = None
                
                for j, s in enumerate(line):
                    if j == base_idx:
                        base_symbol = s
                        break
                
                if base_symbol:
                    # Add superscript element
                    elements.append({
                        "type": "superscript",
                        "base": {
                            "type": "symbol",
                            "value": base_symbol["text"]
                        },
                        "exponent": {
                            "type": "symbol",
                            "value": symbol["text"]
                        }
                    })
                    
                    # Skip the base symbol when we reach it
                    if i < base_idx:
                        line[base_idx]["processed"] = True
                    
            elif relation == "subscript" and "related_to" in symbol:
                # Find the base symbol
                base_idx = symbol["related_to"]
                base_symbol = None
                
                for j, s in enumerate(line):
                    if j == base_idx:
                        base_symbol = s
                        break
                
                if base_symbol:
                    # Add subscript element
                    elements.append({
                        "type": "subscript",
                        "base": {
                            "type": "symbol",
                            "value": base_symbol["text"]
                        },
                        "subscript": {
                            "type": "symbol",
                            "value": symbol["text"]
                        }
                    })
                    
                    # Skip the base symbol when we reach it
                    if i < base_idx:
                        line[base_idx]["processed"] = True
                    
            elif relation in ["numerator", "denominator"] and "fraction_line" in symbol:
                # Collect all numerator and denominator symbols for this fraction
                fraction_line = symbol["fraction_line"]
                numerator_symbols = []
                denominator_symbols = []
                
                for j, s in enumerate(line):
                    if "relation" in s and "fraction_line" in s and s["fraction_line"] == fraction_line:
                        if s["relation"] == "numerator":
                            numerator_symbols.append(s)
                        elif s["relation"] == "denominator":
                            denominator_symbols.append(s)
                        
                        # Mark as processed
                        s["processed"] = True
                
                # Sort numerator and denominator symbols by x position
                numerator_symbols.sort(key=lambda s: s["position"][0])
                denominator_symbols.sort(key=lambda s: s["position"][0])
                
                # Create fraction element
                if numerator_symbols and denominator_symbols:
                    numerator_value = "".join(s["text"] for s in numerator_symbols)
                    denominator_value = "".join(s["text"] for s in denominator_symbols)
                    
                    elements.append({
                        "type": "fraction",
                        "numerator": {
                            "type": "expression",
                            "value": numerator_value
                        },
                        "denominator": {
                            "type": "expression",
                            "value": denominator_value
                        }
                    })
        
        # If not already processed, add as a simple symbol
        if not symbol.get("processed", False):
            elements.append({
                "type": "symbol",
                "value": symbol["text"]
            })
        
        i += 1
    
    # Process operator groupings (like +, -, *, /)
    processed_elements = group_by_operators(elements)
    
    return {
        "type": "expression",
        "elements": processed_elements
    }

def group_by_operators(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group elements by mathematical operators.
    
    Args:
        elements: List of expression elements
        
    Returns:
        List of grouped elements
    """
    # Define operator precedence
    operators = {
        "+": 1, "-": 1,  # Addition and subtraction
        "*": 2, "/": 2,  # Multiplication and division
        "^": 3           # Exponentiation
    }
    
    # Simple parsing for now - in a real implementation this would be more sophisticated
    return elements

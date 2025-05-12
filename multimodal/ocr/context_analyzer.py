import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import latex

logger = logging.getLogger(__name__)

class MathContextAnalyzer:
    """
    Improves OCR recognition quality by using mathematical context
    to disambiguate and correct recognition errors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the math context analyzer.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Common errors in math OCR and their corrections
        self.common_errors = {
            # Digits and letters
            '0': ['O', 'o'],
            '1': ['l', 'I'],
            '2': ['Z', 'z'],
            '5': ['S', 's'],
            '8': ['B'],
            # Operators
            '+': ['t', 'T', '†'],
            '-': ['–', '—', '−'],
            '=': ['≡', '≈'],
            # Greek letters
            'alpha': ['a', 'α'],
            'beta': ['b', 'β'],
            'theta': ['0', 'θ'],
            'pi': ['π', 'n'],
            # ...more mappings
        }
        
        # Build reverse lookup
        self.possible_corrections = {}
        for correct, errors in self.common_errors.items():
            for error in errors:
                self.possible_corrections[error] = correct
        
        # Load common mathematical expressions and patterns
        self._load_math_patterns()
        
        logger.info("Initialized MathContextAnalyzer")
    
    def _load_math_patterns(self):
        """Load common mathematical expressions and patterns."""
        # Common mathematical formulas and theorems
        self.common_expressions = {
            # Basic formulas
            "quadratic": ["ax^2+bx+c=0", "x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}"],
            "pythagorean": ["a^2+b^2=c^2"],
            "euler": ["e^{i\\pi}+1=0"],
            # Calculus
            "derivative_sin": ["\\frac{d}{dx}\\sin(x)=\\cos(x)"],
            "derivative_cos": ["\\frac{d}{dx}\\cos(x)=-\\sin(x)"],
            "integral_x": ["\\int x\\,dx=\\frac{x^2}{2}+C"],
            # Linear algebra
            "matrix_determinant": ["\\det(A)=ad-bc"],
            # Statistics
            "normal_distribution": ["f(x)=\\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}"]
        }
        
        # Common variable naming conventions
        self.variable_patterns = {
            "algebra": ["x", "y", "z", "a", "b", "c", "n", "m"],
            "calculus": ["x", "y", "f", "g", "h", "\\frac{d}{dx}"],
            "geometry": ["\\alpha", "\\beta", "\\theta", "r", "d"],
            "linear_algebra": ["A", "B", "X", "v", "\\lambda"],
            "statistics": ["\\mu", "\\sigma", "X", "P", "E"]
        }
    
    def analyze_and_correct(self, 
                           recognized_symbols: List[Dict[str, Any]],
                           domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze mathematical context and correct recognition errors.
        
        Args:
            recognized_symbols: List of recognized symbols with positions
            domain: Optional mathematical domain for context-specific corrections
            
        Returns:
            Corrected list of symbols with updated confidence scores
        """
        if not recognized_symbols:
            return []
        
        # Step 1: Identify potential domain if not provided
        if domain is None:
            domain = self._identify_domain(recognized_symbols)
            logger.info(f"Identified mathematical domain: {domain}")
        
        # Step 2: Apply domain-specific corrections
        corrected_symbols = self._apply_domain_corrections(recognized_symbols, domain)
        
        # Step 3: Analyze for mathematical coherence
        coherent_symbols = self._ensure_mathematical_coherence(corrected_symbols)
        
        # Step 4: Disambiguate similar symbols
        final_symbols = self._disambiguate_symbols(coherent_symbols)
        
        # Step 5: Perform final validation
        validated_symbols = self._validate_expression(final_symbols)
        
        return validated_symbols
    
    def _identify_domain(self, symbols: List[Dict[str, Any]]) -> str:
        """
        Identify the mathematical domain based on recognized symbols.
        
        Args:
            symbols: List of recognized symbols
            
        Returns:
            Identified mathematical domain
        """
        # Extract symbol texts
        texts = [sym['text'] for sym in symbols]
        joined_text = ' '.join(texts)
        
        # Domain identification heuristics
        domains = {
            "calculus": ["\\int", "\\frac{d}{dx}", "\\sum", "\\lim", "dx", "dy"],
            "linear_algebra": ["matrix", "\\det", "\\begin{pmatrix}", "\\vec"],
            "geometry": ["\\angle", "\\triangle", "\\circle", "\\pi", "\\sin", "\\cos", "\\tan"],
            "statistics": ["\\mu", "\\sigma", "P(", "E[", "Var"],
            "algebra": ["=", "x", "y", "solve"]
        }
        
        # Count domain indicators
        domain_scores = {}
        for domain, indicators in domains.items():
            score = sum(1 for ind in indicators if ind in joined_text)
            domain_scores[domain] = score
        
        # Return the domain with highest score, defaulting to "algebra"
        max_domain = max(domain_scores.items(), key=lambda x: x[1])
        if max_domain[1] > 0:
            return max_domain[0]
        
        return "algebra"  # Default domain
    
    def _apply_domain_corrections(self, 
                                symbols: List[Dict[str, Any]], 
                                domain: str) -> List[Dict[str, Any]]:
        """
        Apply domain-specific corrections to recognized symbols.
        
        Args:
            symbols: List of recognized symbols
            domain: Mathematical domain
            
        Returns:
            Corrected symbols
        """
        corrected = []
        
        # Get common variables for the domain
        common_vars = self.variable_patterns.get(domain, self.variable_patterns["algebra"])
        
        for symbol in symbols:
            text = symbol['text']
            confidence = symbol['confidence']
            
            # Apply domain-specific corrections
            if domain == "calculus":
                # Common calculus-specific corrections
                if text == "a" and "dx" in [s['text'] for s in symbols]:
                    # Likely a differential 'd'
                    corrected.append({**symbol, "text": "d", "confidence": confidence})
                    continue
            
            elif domain == "linear_algebra":
                # Common linear algebra corrections
                if text.lower() == "det" or text.lower() == "det.":
                    corrected.append({**symbol, "text": "\\det", "confidence": confidence})
                    continue
            
            # Apply general corrections based on common errors
            if text in self.possible_corrections:
                # Check if the correction makes sense in this domain
                correction = self.possible_corrections[text]
                if (correction in common_vars or 
                    any(correction in expr for expr in self.common_expressions.get(domain, []))):
                    corrected.append({
                        **symbol, 
                        "text": correction,
                        "original_text": text,
                        "confidence": confidence * 1.1  # Slight boost in confidence
                    })
                    continue
            
            # No correction needed
            corrected.append(symbol)
        
        return corrected
    
    def _ensure_mathematical_coherence(self, symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure mathematical coherence by checking equation structure.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Coherent symbols with potential corrections
        """
        if not symbols:
            return []
        
        # Check parentheses matching
        left_parens = sum(1 for s in symbols if s['text'] in ['(', '[', '{', '\\{', '\\left('])
        right_parens = sum(1 for s in symbols if s['text'] in [')', ']', '}', '\\}', '\\right)'])
        
        if left_parens > right_parens:
            # Missing right parenthesis, look for potential misrecognitions
            for i, symbol in enumerate(symbols):
                if symbol['text'] in ['c', 'C', 'o', 'O'] and symbol['confidence'] < 0.8:
                    # Possible misrecognized right parenthesis
                    symbols[i] = {**symbol, "text": ")", "confidence": symbol['confidence'] * 0.9}
                    right_parens += 1
                    if left_parens == right_parens:
                        break
        
        elif right_parens > left_parens:
            # Missing left parenthesis, look for potential misrecognitions
            for i, symbol in enumerate(symbols):
                if symbol['text'] in ['c', 'C', 'o', 'O'] and symbol['confidence'] < 0.8:
                    # Possible misrecognized left parenthesis
                    symbols[i] = {**symbol, "text": "(", "confidence": symbol['confidence'] * 0.9}
                    left_parens += 1
                    if left_parens == right_parens:
                        break
        
        # Check operator-operand patterns
        for i in range(len(symbols) - 1):
            current, next_sym = symbols[i], symbols[i+1]
            
            # Two operators in a row - possible error
            if (current['text'] in ['+', '-', '*', '/', '\\times', '\\div'] and 
                next_sym['text'] in ['+', '*', '/', '\\times', '\\div']):
                # Lower confidence for the second operator
                symbols[i+1] = {**next_sym, "confidence": next_sym['confidence'] * 0.7}
        
        return symbols
    
    def _disambiguate_symbols(self, symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Disambiguate similar-looking symbols based on context.
        
        Args:
            symbols: List of symbols to disambiguate
            
        Returns:
            Disambiguated symbols
        """
        result = []
        
        for i, symbol in enumerate(symbols):
            text = symbol['text']
            confidence = symbol['confidence']
            
            # Previous and next symbols for context (if available)
            prev_sym = symbols[i-1]['text'] if i > 0 else None
            next_sym = symbols[i+1]['text'] if i < len(symbols) - 1 else None
            
            # Disambiguate based on common patterns
            # 0/O confusion
            if text in ['0', 'O', 'o']:
                if prev_sym and prev_sym.isalpha():
                    # Likely a variable subscript - prefer 0
                    result.append({**symbol, "text": "0", "confidence": max(confidence, 0.8)})
                elif i > 0 and i < len(symbols) - 1 and symbols[i-1]['text'].isdigit() and symbols[i+1]['text'].isdigit():
                    # Part of a number - prefer 0
                    result.append({**symbol, "text": "0", "confidence": max(confidence, 0.9)})
                else:
                    # Likely a variable - prefer O
                    result.append({**symbol, "text": "O", "confidence": max(confidence, 0.7)})
                continue
            
            # l/1/I confusion
            if text in ['l', '1', 'I']:
                if i > 0 and symbols[i-1]['text'].isdigit():
                    # Part of a number - prefer 1
                    result.append({**symbol, "text": "1", "confidence": max(confidence, 0.9)})
                elif next_sym and next_sym in ['n', 'm', ')', ']', '+', '-', '*', '/']:
                    # Common variable pattern - prefer l
                    result.append({**symbol, "text": "l", "confidence": max(confidence, 0.8)})
                else:
                    # Prefer I for isolated cases
                    result.append({**symbol, "text": "I", "confidence": max(confidence, 0.7)})
                continue
            
            # Default: keep as is
            result.append(symbol)
        
        return result
    
    def _validate_expression(self, symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate that the expression is mathematically valid.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Validated and potentially corrected symbols
        """
        if not symbols:
            return []
        
        # Build LaTeX representation
        latex_str = ""
        for symbol in symbols:
            latex_str += symbol['text']
        
        # Basic validation checks
        validated = symbols.copy()
        
        try:
            # Try to parse with SymPy for syntactical validity
            # Note: This is challenging with LaTeX, so we do basic checks
            
            # Check for unbalanced environments
            if "\\begin{" in latex_str:
                begin_count = latex_str.count("\\begin{")
                end_count = latex_str.count("\\end{")
                
                if begin_count != end_count:
                    logger.warning(f"Unbalanced LaTeX environments: {begin_count} begins, {end_count} ends")
            
            # Check for basic equation structure
            if "=" in latex_str:
                parts = latex_str.split("=")
                if len(parts) > 2:
                    # Multiple equal signs - could be correct in some contexts
                    logger.info(f"Multiple equal signs detected: {latex_str}")
            
        except Exception as e:
            logger.warning(f"Expression validation error: {e}")
        
        return validated

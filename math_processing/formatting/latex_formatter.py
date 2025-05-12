"""
Advanced LaTeX Formatting Module

This module provides comprehensive LaTeX formatting capabilities for mathematical expressions,
focusing on typographical optimization, consistent formatting standards, and specialized
domain-specific formatting.
"""

import re
import logging
from typing import Dict, List, Optional, Union, Any

import sympy as sp
from sympy.printing.latex import LatexPrinter

logger = logging.getLogger(__name__)

class LatexFormatter:
    """
    Advanced LaTeX formatter for mathematical expressions.
    
    This class provides utilities to format LaTeX expressions with proper
    spacing, structure, and typographical conventions for optimal readability
    and aesthetic presentation.
    """
    
    def __init__(self, formatting_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LaTeX formatter.
        
        Args:
            formatting_config: Optional configuration dictionary with formatting preferences
        """
        self.config = formatting_config or self._default_config()
        self.custom_printer = CustomLatexPrinter(settings=self.config.get("printer_settings", {}))
        
    def format_expression(self, expression: Union[str, sp.Expr]) -> str:
        """
        Format a LaTeX expression with enhanced typographical features.
        
        Args:
            expression: LaTeX string or SymPy expression to format
            
        Returns:
            Formatted LaTeX string
        """
        # Convert SymPy expression to LaTeX if needed
        if isinstance(expression, sp.Expr):
            latex_expr = self.custom_printer.doprint(expression)
        else:
            latex_expr = expression
            
        # Apply formatting enhancements
        latex_expr = self._standardize_spacing(latex_expr)
        latex_expr = self._fix_fence_sizing(latex_expr)
        latex_expr = self._enhance_fractions(latex_expr)
        latex_expr = self._optimize_supersubs(latex_expr)
        
        # Apply domain-specific formatting if detected
        domain = self._detect_domain(latex_expr)
        if domain:
            latex_expr = self._apply_domain_formatting(latex_expr, domain)
            
        return latex_expr
    
    def format_step_solution(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format a step-by-step solution with consistent LaTeX formatting.
        
        Args:
            steps: List of solution steps with LaTeX expressions
            
        Returns:
            List of steps with formatted LaTeX
        """
        formatted_steps = []
        
        for step in steps:
            formatted_step = step.copy()
            
            # Format the input expression if present
            if 'input' in step and step['input']:
                if isinstance(step['input'], str):
                    formatted_step['input'] = self.format_expression(step['input'])
                    
            # Format the output expression if present
            if 'output' in step and step['output']:
                if isinstance(step['output'], str):
                    formatted_step['output'] = self.format_expression(step['output'])
            
            # Format any additional expressions in the step
            if 'latex_expressions' in step and step['latex_expressions']:
                if isinstance(step['latex_expressions'], list):
                    formatted_step['latex_expressions'] = [
                        self.format_expression(expr) for expr in step['latex_expressions']
                    ]
                elif isinstance(step['latex_expressions'], str):
                    formatted_step['latex_expressions'] = self.format_expression(step['latex_expressions'])
            
            formatted_steps.append(formatted_step)
            
        return formatted_steps
    
    def create_equation_environment(self, expressions: Union[str, List[str]], 
                                  numbered: bool = False,
                                  aligned: bool = False) -> str:
        """
        Create a proper LaTeX equation environment.
        
        Args:
            expressions: LaTeX expression(s) to include in the environment
            numbered: Whether to include equation numbers
            aligned: Whether to use aligned environment for multiple equations
            
        Returns:
            LaTeX string with proper equation environment
        """
        if isinstance(expressions, str):
            expressions = [expressions]
            
        # Format all expressions
        formatted_exprs = [self.format_expression(expr) for expr in expressions]
        
        # Select the appropriate environment
        if numbered:
            env_begin = r"\begin{equation}" if len(formatted_exprs) == 1 else r"\begin{align}"
            env_end = r"\end{equation}" if len(formatted_exprs) == 1 else r"\end{align}"
        else:
            env_begin = r"\begin{equation*}" if len(formatted_exprs) == 1 else r"\begin{align*}"
            env_end = r"\end{equation*}" if len(formatted_exprs) == 1 else r"\end{align*}"
        
        # Build the environment content
        if len(formatted_exprs) == 1:
            content = formatted_exprs[0]
        else:
            if aligned:
                lines = []
                for expr in formatted_exprs:
                    if "&" not in expr:
                        # Add alignment point at equals sign if not already present
                        expr = expr.replace("=", "&=")
                    lines.append(expr)
                content = r" \\ ".join(lines)
            else:
                content = r" \\ ".join(formatted_exprs)
        
        # Assemble the complete environment
        return f"{env_begin}\n{content}\n{env_end}"
    
    def create_display_math(self, expression: str) -> str:
        """
        Create a display math environment using dollar delimiters.
        
        Args:
            expression: LaTeX expression to format
            
        Returns:
            Formatted display math LaTeX
        """
        formatted_expr = self.format_expression(expression)
        return f"$${formatted_expr}$$"
    
    def create_inline_math(self, expression: str) -> str:
        """
        Create an inline math environment using single dollar delimiters.
        
        Args:
            expression: LaTeX expression to format
            
        Returns:
            Formatted inline math LaTeX
        """
        formatted_expr = self.format_expression(expression)
        return f"${formatted_expr}$"
    
    def optimize_for_display(self, latex_expr: str, display_context: str = "web") -> str:
        """
        Optimize LaTeX for specific display contexts.
        
        Args:
            latex_expr: LaTeX expression to optimize
            display_context: Target context ("web", "print", "presentation")
            
        Returns:
            Optimized LaTeX expression
        """
        if display_context == "web":
            # Optimize for web display (MathJax/KaTeX)
            # Avoid excessive nesting that might cause rendering issues
            latex_expr = self._reduce_nesting(latex_expr)
            
            # Use simpler notation for common functions
            latex_expr = self._simplify_common_functions(latex_expr)
            
        elif display_context == "print":
            # Optimize for print layout
            # Use higher resolution symbols
            latex_expr = self._enhance_for_print(latex_expr)
            
        elif display_context == "presentation":
            # Optimize for presentations
            # Use larger and more visible elements
            latex_expr = self._enhance_for_presentation(latex_expr)
            
        return latex_expr
    
    def format_matrix(self, matrix: Union[List[List[Any]], sp.Matrix], 
                    bracket_type: str = "bracket",
                    alignment: str = "c") -> str:
        """
        Format a matrix with proper LaTeX conventions.
        
        Args:
            matrix: 2D list or SymPy matrix
            bracket_type: Type of brackets ("bracket", "paren", "vert", "brace")
            alignment: Column alignment ("c", "l", "r")
            
        Returns:
            Formatted LaTeX matrix
        """
        # Convert SymPy matrix to list if needed
        if isinstance(matrix, sp.Matrix):
            matrix_data = matrix.tolist()
        else:
            matrix_data = matrix
            
        # Determine matrix environment based on bracket type
        if bracket_type == "bracket":
            env = "bmatrix"
        elif bracket_type == "paren":
            env = "pmatrix" 
        elif bracket_type == "vert":
            env = "vmatrix"
        elif bracket_type == "brace":
            env = "Bmatrix"
        else:
            env = "matrix"  # No brackets
            
        # Create the matrix environment
        result = [f"\\begin{{{env}}}"]
        
        # Add rows with proper alignment
        for row in matrix_data:
            formatted_row = " & ".join([self.format_expression(str(item)) if not isinstance(item, sp.Expr) else self.format_expression(item) for item in row])
            result.append(formatted_row + r" \\")
            
        # Close the environment
        result.append(f"\\end{{{env}}}")
        
        return "\n".join(result)
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Return default configuration settings.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "printer_settings": {
                "order": "none",
                "full_prec": False,
                "trig_format": "power",
            },
            "spacing": {
                "binary_ops": True,
                "relations": True,
                "fence_content": True
            },
            "fractions": {
                "inline_small_fracs": True,
                "display_frac_style": "displaystyle"
            },
            "fonts": {
                "use_mathcal": True,
                "use_mathfrak": False,
                "use_mathbb": True
            }
        }
    
    def _standardize_spacing(self, latex_expr: str) -> str:
        """
        Standardize spacing around operators and relations.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Formatted LaTeX with improved spacing
        """
        if self.config["spacing"]["binary_ops"]:
            # Add spacing around binary operators (but not in subscripts/superscripts)
            binary_ops = ["+", "-", r"\cdot", r"\times", r"\div"]
            
            for op in binary_ops:
                # Don't add spaces within subscripts or superscripts
                if op in ["+", "-"]:
                    # Special case for + and - which might be unary
                    # Add space only if preceded by non-command and not in sub/super
                    latex_expr = re.sub(r'([^\\^_{ ])(' + re.escape(op) + r')([^}])', r'\1 \2 \3', latex_expr)
                else:
                    # Other binary operators always get spaces
                    latex_expr = re.sub(r'([^_^])(' + re.escape(op) + r')([^}])', r'\1 \2 \3', latex_expr)
        
        if self.config["spacing"]["relations"]:
            # Add spacing around relation symbols
            relations = ["=", r"\approx", r"\sim", r"\simeq", r"\cong", r"\equiv",
                        "<", ">", r"\le", r"\ge", r"\neq", r"\in", r"\subset"]
            
            for rel in relations:
                latex_expr = re.sub(r'([^\\])(' + re.escape(rel) + r')', r'\1 \2 ', latex_expr)
        
        if self.config["spacing"]["fence_content"]:
            # Add small spacing inside fences for better readability
            latex_expr = re.sub(r'(\\left[({[|])(.*?)(\\right[)}\]|])', r'\1\\, \2 \\, \3', latex_expr)
            
        return latex_expr
    
    def _fix_fence_sizing(self, latex_expr: str) -> str:
        """
        Ensure proper sizing of fences (parentheses, brackets, etc.).
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            LaTeX with properly sized fences
        """
        # Replace \big, \Big, etc. with \left and \right for dynamic sizing
        sizing_commands = [r"\big", r"\Big", r"\bigg", r"\Bigg"]
        opening_fences = [r"(", r"[", r"\{", r"|"]
        closing_fences = [r")", r"]", r"\}", r"|"]
        
        for cmd in sizing_commands:
            for fence in opening_fences:
                latex_expr = latex_expr.replace(f"{cmd}{fence}", f"\\left{fence}")
            for fence in closing_fences:
                latex_expr = latex_expr.replace(f"{cmd}{fence}", f"\\right{fence}")
        
        # Ensure paired \left and \right
        # Count left and right commands
        left_count = latex_expr.count(r"\left")
        right_count = latex_expr.count(r"\right")
        
        # Add missing \right or \left if needed
        if left_count > right_count:
            for _ in range(left_count - right_count):
                latex_expr += r"\right."
        elif right_count > left_count:
            for _ in range(right_count - left_count):
                latex_expr = r"\left." + latex_expr
                
        return latex_expr
    
    def _enhance_fractions(self, latex_expr: str) -> str:
        """
        Enhance the formatting of fractions for better readability.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            LaTeX with enhanced fractions
        """
        # Use \dfrac for display-style fractions (larger, more readable)
        if self.config["fractions"]["display_frac_style"] == "displaystyle":
            latex_expr = latex_expr.replace(r"\frac", r"\dfrac")
        
        # For inline fractions, consider converting small fractions to slashed form
        if self.config["fractions"]["inline_small_fracs"]:
            # Convert simple fractions like \frac{1}{2} to slashed form 1/2
            simple_frac_pattern = r"\\frac\{(\d+)\}\{(\d+)\}"
            
            def replace_with_slash(match):
                num, denom = match.groups()
                if len(num) == 1 and len(denom) == 1:
                    return f"{num}/{denom}"
                return match.group(0)
                
            latex_expr = re.sub(simple_frac_pattern, replace_with_slash, latex_expr)
            
        return latex_expr
    
    def _optimize_supersubs(self, latex_expr: str) -> str:
        """
        Optimize superscripts and subscripts for better readability.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            LaTeX with optimized superscripts and subscripts
        """
        # Ensure proper braces for multi-character sub/superscripts
        # Find subscripts/superscripts followed by more than one character without braces
        subscript_pattern = r'_([^{}\s][^{}\s]+)'
        superscript_pattern = r'\^([^{}\s][^{}\s]+)'
        
        # Add braces around multi-character sub/superscripts
        latex_expr = re.sub(subscript_pattern, r'_{\1}', latex_expr)
        latex_expr = re.sub(superscript_pattern, r'^{\1}', latex_expr)
        
        return latex_expr
    
    def _detect_domain(self, latex_expr: str) -> Optional[str]:
        """
        Detect the mathematical domain of an expression.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Detected domain or None
        """
        # Check for calculus indicators
        calculus_indicators = [r"\int", r"\sum", r"\lim", r"\frac{d}{d", r"\partial", r"\nabla"]
        if any(indicator in latex_expr for indicator in calculus_indicators):
            return "calculus"
            
        # Check for linear algebra indicators
        linear_algebra_indicators = [r"\begin{matrix}", r"\begin{pmatrix}", r"\begin{bmatrix}", 
                                    r"\begin{vmatrix}", r"\vec{", r"\mathbf{"]
        if any(indicator in latex_expr for indicator in linear_algebra_indicators):
            return "linear_algebra"
            
        # Check for statistics indicators
        statistics_indicators = [r"\mathbb{P}", r"\mathbb{E}", r"\sigma", r"\mu", r"\bar{x}", 
                               r"\sim", r"\mathcal{N}"]
        if any(indicator in latex_expr for indicator in statistics_indicators):
            return "statistics"
            
        # Check for set theory indicators
        set_theory_indicators = [r"\cup", r"\cap", r"\subset", r"\in", r"\emptyset", r"\mathbb{Z}", 
                               r"\mathbb{R}", r"\mathbb{N}"]
        if any(indicator in latex_expr for indicator in set_theory_indicators):
            return "set_theory"
            
        return None
    
    def _apply_domain_formatting(self, latex_expr: str, domain: str) -> str:
        """
        Apply domain-specific formatting conventions.
        
        Args:
            latex_expr: LaTeX expression
            domain: Mathematical domain
            
        Returns:
            LaTeX with domain-specific formatting
        """
        if domain == "calculus":
            # Ensure proper integral formatting
            latex_expr = self._format_calculus(latex_expr)
            
        elif domain == "linear_algebra":
            # Format vectors and matrices with proper fonts
            latex_expr = self._format_linear_algebra(latex_expr)
            
        elif domain == "statistics":
            # Format statistical notation
            latex_expr = self._format_statistics(latex_expr)
            
        elif domain == "set_theory":
            # Format set notation
            latex_expr = self._format_set_theory(latex_expr)
            
        return latex_expr
    
    def _format_calculus(self, latex_expr: str) -> str:
        """
        Apply calculus-specific formatting.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Formatted calculus expression
        """
        # Improve integral formatting
        # Add proper spacing after integral sign
        latex_expr = re.sub(r'(\\int)([^_\s\\])', r'\1 \2', latex_expr)
        
        # Ensure proper differential formatting
        # Format dx as a proper differential
        latex_expr = re.sub(r'([^\\])d([a-zA-Z])', r'\1\\,\\mathrm{d}\2', latex_expr)
        
        # Ensure proper limits formatting
        latex_expr = re.sub(r'\\lim_', r'\\lim\\limits_', latex_expr)
        
        return latex_expr
    
    def _format_linear_algebra(self, latex_expr: str) -> str:
        """
        Apply linear algebra-specific formatting.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Formatted linear algebra expression
        """
        # Apply proper vector formatting
        # Convert vector notation to bold
        latex_expr = re.sub(r'\\vec{([^}]+)}', r'\\mathbf{\1}', latex_expr)
        
        # Ensure matrices have proper formatting
        # Add small spacing in matrices for readability
        latex_expr = re.sub(r'(&)([^&\\]+)(&)', r'\1\\,\2\\,\3', latex_expr)
        
        return latex_expr
    
    def _format_statistics(self, latex_expr: str) -> str:
        """
        Apply statistics-specific formatting.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Formatted statistics expression
        """
        # Format probability notation
        latex_expr = re.sub(r'Pr\s*\(', r'\\mathbb{P}(', latex_expr)
        latex_expr = re.sub(r'E\s*\[', r'\\mathbb{E}[', latex_expr)
        
        # Format variance and standard deviation
        latex_expr = re.sub(r'Var\s*\(', r'\\operatorname{Var}(', latex_expr)
        latex_expr = re.sub(r'Std\s*\(', r'\\operatorname{Std}(', latex_expr)
        
        return latex_expr
    
    def _format_set_theory(self, latex_expr: str) -> str:
        """
        Apply set theory-specific formatting.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Formatted set theory expression
        """
        # Format set notation
        # Ensure proper set notation with \mathbb
        common_sets = {
            r'\\mathbb{R}': r'\\mathbb{R}',  # Reals
            r'\\mathbb{Z}': r'\\mathbb{Z}',  # Integers
            r'\\mathbb{N}': r'\\mathbb{N}',  # Natural numbers
            r'\\mathbb{Q}': r'\\mathbb{Q}',  # Rationals
            r'\\mathbb{C}': r'\\mathbb{C}',  # Complex numbers
        }
        
        for pattern, replacement in common_sets.items():
            # Replace non-mathbb versions with mathbb
            non_mathbb = pattern.replace(r'\mathbb{', '').replace('}', '')
            latex_expr = re.sub(r'([^\\])' + non_mathbb, r'\1' + pattern, latex_expr)
            
        # Format set operations with proper spacing
        set_ops = [r'\cup', r'\cap', r'\setminus', r'\triangle', r'\oplus']
        for op in set_ops:
            latex_expr = re.sub(r'([^\\])(' + re.escape(op) + r')([^}])', r'\1 \2 \3', latex_expr)
            
        return latex_expr
    
    def _reduce_nesting(self, latex_expr: str) -> str:
        """
        Reduce excessive nesting in LaTeX for better rendering.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            LaTeX with reduced nesting
        """
        # Simplify nested fractions when possible
        # Match nested fractions pattern
        nested_frac_pattern = r'\\frac\{([^{}]+)\}\{\\frac\{([^{}]+)\}\{([^{}]+)\}\}'
        
        def simplify_nested_frac(match):
            num = match.group(1)
            inner_num = match.group(2)
            inner_denom = match.group(3)
            return fr'\\frac{{{num} \\cdot {inner_denom}}}{{{inner_num}}}'
            
        # Apply simplification
        latex_expr = re.sub(nested_frac_pattern, simplify_nested_frac, latex_expr)
        
        return latex_expr
    
    def _simplify_common_functions(self, latex_expr: str) -> str:
        """
        Simplify common function notations for better web rendering.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Simplified LaTeX expression
        """
        # Simplify common trigonometric functions
        trig_funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 
                     'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']
                     
        for func in trig_funcs:
            # Ensure proper function formatting with \
            latex_expr = re.sub(r'([^\\])' + func + r'([^a-zA-Z])', r'\1\\' + func + r'\2', latex_expr)
            
        # Simplify other common functions
        other_funcs = ['log', 'ln', 'exp', 'lim', 'sup', 'inf', 'min', 'max']
        
        for func in other_funcs:
            # Ensure proper function formatting with \
            latex_expr = re.sub(r'([^\\])' + func + r'([^a-zA-Z])', r'\1\\' + func + r'\2', latex_expr)
            
        return latex_expr
    
    def _enhance_for_print(self, latex_expr: str) -> str:
        """
        Enhance LaTeX for print rendering.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Print-optimized LaTeX expression
        """
        # Use high-resolution symbols
        latex_expr = latex_expr.replace(r"\sum", r"\sum\limits")
        latex_expr = latex_expr.replace(r"\prod", r"\prod\limits")
        
        # Use displaystyle for fractions
        latex_expr = latex_expr.replace(r"\frac", r"\displaystyle\frac")
        
        return latex_expr
    
    def _enhance_for_presentation(self, latex_expr: str) -> str:
        """
        Enhance LaTeX for presentation slides.
        
        Args:
            latex_expr: LaTeX expression
            
        Returns:
            Presentation-optimized LaTeX expression
        """
        # Use larger and more visible elements
        latex_expr = latex_expr.replace(r"\int", r"\displaystyle\int")
        latex_expr = latex_expr.replace(r"\sum", r"\displaystyle\sum")
        latex_expr = latex_expr.replace(r"\frac", r"\dfrac")
        
        # Increase spacing for readability from a distance
        latex_expr = re.sub(r'([=<>+\-])([^=<>+\-])', r'\1\\;\\;\2', latex_expr)
        
        return latex_expr


class CustomLatexPrinter(LatexPrinter):
    """
    Extended LaTeX printer with enhanced formatting options.
    """
    
    def __init__(self, settings=None):
        """Initialize with custom settings."""
        settings = settings or {}
        super().__init__(settings)
    
    def _print_Pow(self, expr):
        """Enhanced power printing for better readability."""
        # Special case for square roots
        if expr.exp == sp.Rational(1, 2):
            return r"\sqrt{%s}" % self._print(expr.base)
        elif expr.exp == sp.Rational(-1, 2):
            return r"\frac{1}{\sqrt{%s}}" % self._print(expr.base)
            
        # Special case for cubic roots
        elif expr.exp == sp.Rational(1, 3):
            return r"\sqrt[3]{%s}" % self._print(expr.base)
            
        # Handle other cases
        return super()._print_Pow(expr)
    
    def _print_Mul(self, expr):
        """Enhanced multiplication printing."""
        # Use \cdot for multiplication with better spacing
        from sympy.core.mul import _keep_coeff
        from sympy.core import Symbol
        
        # Handle special cases like coefficient * function
        if len(expr.args) == 2 and isinstance(expr.args[1], sp.Function):
            # Handle coefficient * function case
            if isinstance(expr.args[0], sp.Number):
                if expr.args[0] == 1:
                    return self._print(expr.args[1])
                elif expr.args[0] == -1:
                    return r"-" + self._print(expr.args[1])
                else:
                    return self._print(expr.args[0]) + r" \cdot " + self._print(expr.args[1])
                    
        return super()._print_Mul(expr)

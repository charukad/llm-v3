"""
Explainer - provides natural language explanations for mathematical steps.

This module generates detailed textual explanations for mathematical solution steps,
making them more understandable for users.
"""

import sympy as sp
from typing import Dict, List, Union, Any, Optional
import logging
import re


class StepExplainer:
    """Explainer for mathematical solution steps."""
    
    def __init__(self):
        """Initialize the step explainer."""
        self.logger = logging.getLogger(__name__)
        self.explanation_templates = self._load_explanation_templates()
    
    def explain_steps(self, 
                    steps: List[Dict[str, str]], 
                    operation: str, 
                    context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """
        Generate natural language explanations for solution steps.
        
        Args:
            steps: List of solution steps (each with expression and basic explanation)
            operation: Type of operation being explained
            context: Additional context for explanations
            
        Returns:
            List of steps with enhanced explanations
        """
        try:
            # Initialize context if None
            if context is None:
                context = {}
            
            # Get the appropriate explainer for the operation
            if operation.lower() == "solve":
                return self._explain_solve_steps(steps, context)
            elif operation.lower() == "differentiate":
                return self._explain_differentiation_steps(steps, context)
            elif operation.lower() == "integrate":
                return self._explain_integration_steps(steps, context)
            elif operation.lower() == "limit":
                return self._explain_limit_steps(steps, context)
            elif operation.lower() == "factor":
                return self._explain_factoring_steps(steps, context)
            elif operation.lower() == "expand":
                return self._explain_expansion_steps(steps, context)
            elif operation.lower() == "simplify":
                return self._explain_simplification_steps(steps, context)
            else:
                # For unsupported operations, return the original steps
                return steps
        except Exception as e:
            self.logger.error(f"Error explaining steps: {str(e)}")
            # Return the original steps if an error occurs
            return steps
    
    def _load_explanation_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load explanation templates for different operations and steps.
        
        Returns:
            Dictionary of templates organized by operation and step type
        """
        # This would typically load from a file or database in a real implementation
        # Here we define them inline for simplicity
        
        templates = {
            "solve": {
                "equation_setup": [
                    "We start by writing the equation in standard form.",
                    "First, let's examine the equation we need to solve.",
                    "Our goal is to solve this equation for the variable."
                ],
                "isolate_variable": [
                    "We isolate the variable by moving all other terms to the opposite side.",
                    "To solve for the variable, we need to isolate it on one side of the equation.",
                    "We rearrange the equation to get the variable by itself on one side."
                ],
                "simplify": [
                    "We simplify the expression to make it easier to work with.",
                    "Let's simplify this expression by combining like terms.",
                    "Simplifying this expression will make the next steps clearer."
                ],
                "quadratic_formula": [
                    "Since this is a quadratic equation in the form ax² + bx + c = 0, we can use the quadratic formula: x = (-b ± √(b² - 4ac))/2a.",
                    "For this quadratic equation, we'll apply the quadratic formula to find the solutions.",
                    "The quadratic formula gives us a direct way to find the solutions of this equation."
                ],
                "discriminant": [
                    "The discriminant b² - 4ac tells us about the nature of the solutions.",
                    "We calculate the discriminant to determine how many real solutions exist.",
                    "Computing the discriminant will tell us whether we have real or complex solutions."
                ],
                "solution": [
                    "Therefore, the solution to the equation is as shown.",
                    "This gives us our final solution.",
                    "We've found the value(s) of the variable that satisfy the original equation."
                ]
            },
            "differentiate": {
                "initial": [
                    "We need to find the derivative of this expression.",
                    "Our goal is to differentiate this function with respect to the variable.",
                    "Let's apply the rules of differentiation to this expression."
                ],
                "power_rule": [
                    "We use the power rule: d/dx[x^n] = n·x^(n-1).",
                    "Applying the power rule for differentiation.",
                    "The power rule tells us that the derivative of x^n is n·x^(n-1)."
                ],
                "product_rule": [
                    "We use the product rule: d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x).",
                    "Since this is a product of functions, we apply the product rule.",
                    "The product rule allows us to differentiate this product of functions."
                ],
                "quotient_rule": [
                    "We use the quotient rule: d/dx[f(x)/g(x)] = [f'(x)·g(x) - f(x)·g'(x)]/[g(x)]².",
                    "For this quotient, we apply the quotient rule of differentiation.",
                    "The quotient rule is needed to differentiate this fraction."
                ],
                "chain_rule": [
                    "We apply the chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x).",
                    "Since we have a composite function, we need to use the chain rule.",
                    "The chain rule allows us to differentiate this composition of functions."
                ],
                "final": [
                    "After applying the differentiation rules, we get the derivative.",
                    "This gives us the final derivative of the original expression.",
                    "We have now found the derivative of the given function."
                ]
            },
            "integrate": {
                "initial": [
                    "We need to find the integral of this expression.",
                    "Our goal is to integrate this function with respect to the variable.",
                    "Let's apply the rules of integration to this expression."
                ],
                "power_rule": [
                    "We use the power rule for integration: ∫x^n dx = x^(n+1)/(n+1) + C, for n≠-1.",
                    "Applying the power rule for integration.",
                    "The power rule tells us that the integral of x^n is x^(n+1)/(n+1) + C."
                ],
                "substitution": [
                    "We use substitution (u-substitution) to simplify this integral.",
                    "Let's apply the substitution method to transform this integral.",
                    "Substitution helps us convert this integral into a more manageable form."
                ],
                "parts": [
                    "We use integration by parts: ∫u·dv = u·v - ∫v·du.",
                    "For this type of integral, integration by parts is appropriate.",
                    "Integration by parts allows us to handle this product of functions."
                ],
                "definite": [
                    "For a definite integral, we evaluate the antiderivative at the upper and lower bounds and subtract.",
                    "To compute this definite integral, we find the antiderivative and then evaluate it at the bounds.",
                    "The Fundamental Theorem of Calculus tells us to evaluate the antiderivative at the bounds."
                ],
                "final": [
                    "After applying the integration rules, we get the result.",
                    "This gives us the final integral of the original expression.",
                    "We have now found the integral of the given function."
                ]
            },
            "limit": {
                "initial": [
                    "We need to find the limit of this expression as the variable approaches the given value.",
                    "Our goal is to evaluate this limit.",
                    "Let's determine what happens to this expression as the variable approaches the specified point."
                ],
                "direct_substitution": [
                    "We can evaluate this limit by direct substitution.",
                    "In this case, direct substitution gives us the limit.",
                    "Since there's no indeterminate form, we can substitute the value directly."
                ],
                "indeterminate": [
                    "Direct substitution gives an indeterminate form, so we need other techniques.",
                    "We get an indeterminate form, which requires further analysis.",
                    "This results in an indeterminate form that we need to resolve."
                ],
                "factoring": [
                    "We can resolve this indeterminate form by factoring and cancellation.",
                    "Factoring helps us eliminate the common factor causing the indeterminacy.",
                    "By factoring, we can simplify the expression before taking the limit."
                ],
                "lhopital": [
                    "We apply L'Hôpital's rule, which states that for indeterminate forms 0/0 or ∞/∞, the limit equals the limit of the ratio of derivatives.",
                    "L'Hôpital's rule lets us replace this limit with the limit of the ratio of derivatives.",
                    "Since we have an indeterminate form, L'Hôpital's rule is applicable."
                ],
                "final": [
                    "After applying the appropriate techniques, we find the limit.",
                    "This gives us the final value of the limit.",
                    "We have now determined the limit of the given expression."
                ]
            },
            # Add templates for other operations...
        }
        
        return templates
    
    def _explain_solve_steps(self, 
                           steps: List[Dict[str, str]], 
                           context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate explanations for equation solving steps.
        
        Args:
            steps: List of solution steps
            context: Additional context
            
        Returns:
            List of steps with enhanced explanations
        """
        # Make a copy of the steps to avoid modifying the original
        enhanced_steps = steps.copy()
        
        # Extract variable information for better explanations
        variable = context.get("variable")
        if isinstance(variable, str):
            var_name = variable
        elif isinstance(variable, sp.Symbol):
            var_name = variable.name
        else:
            var_name = "x"  # Default variable name
        
        # Enhance explanations based on step content
        for i, step in enumerate(enhanced_steps):
            current_explanation = step["explanation"]
            expression = step["expression"]
            
            # Check for different types of steps and enhance explanations
            if "equation" in current_explanation.lower() and "original" in current_explanation.lower():
                templates = self.explanation_templates["solve"]["equation_setup"]
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(
                    "We want to solve for " + var_name + ".", context
                )
            
            elif "quadratic" in current_explanation.lower() and "formula" in current_explanation.lower():
                templates = self.explanation_templates["solve"]["quadratic_formula"]
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(
                    "This will give us the solutions directly.", context
                )
            
            elif "discriminant" in current_explanation.lower():
                templates = self.explanation_templates["solve"]["discriminant"]
                
                # Check if we can determine the discriminant value from the expression
                discriminant_match = re.search(r"\\Delta = .* = ([-\d.]+)", expression)
                if discriminant_match:
                    discriminant_value = float(discriminant_match.group(1))
                    if discriminant_value > 0:
                        detail = "Since the discriminant is positive, we'll have two distinct real solutions."
                    elif discriminant_value == 0:
                        detail = "Since the discriminant is zero, we'll have exactly one real solution (a repeated root)."
                    else:
                        detail = "Since the discriminant is negative, we'll have two complex conjugate solutions."
                    
                    step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(detail, context)
                else:
                    step["explanation"] = self._select_template(templates)
            
            elif "solution" in current_explanation.lower() and i == len(steps) - 1:
                templates = self.explanation_templates["solve"]["solution"]
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(
                    "These values of " + var_name + " satisfy the original equation.", context
                )
        
        return enhanced_steps
    
    def _explain_differentiation_steps(self, 
                                     steps: List[Dict[str, str]], 
                                     context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate explanations for differentiation steps.
        
        Args:
            steps: List of differentiation steps
            context: Additional context
            
        Returns:
            List of steps with enhanced explanations
        """
        # Make a copy of the steps to avoid modifying the original
        enhanced_steps = steps.copy()
        
        # Extract variable information for better explanations
        variable = context.get("variable")
        if isinstance(variable, str):
            var_name = variable
        elif isinstance(variable, sp.Symbol):
            var_name = variable.name
        else:
            var_name = "x"  # Default variable name
        
        order = context.get("order", 1)
        
        # Enhance explanations based on step content
        for i, step in enumerate(enhanced_steps):
            current_explanation = step["explanation"]
            expression = step["expression"]
            
            # Check for different types of steps and enhance explanations
            if i == 0:  # First step (original expression)
                templates = self.explanation_templates["differentiate"]["initial"]
                
                if order > 1:
                    detail = f"We need to find the {self._ordinal(order)} derivative with respect to {var_name}."
                else:
                    detail = f"We need to find the derivative with respect to {var_name}."
                
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(detail, context)
            
            elif "power rule" in current_explanation.lower() or "x^" in expression:
                templates = self.explanation_templates["differentiate"]["power_rule"]
                step["explanation"] = self._select_template(templates)
            
            elif "product" in current_explanation.lower() or "·" in expression or "*" in expression:
                templates = self.explanation_templates["differentiate"]["product_rule"]
                step["explanation"] = self._select_template(templates)
            
            elif "quotient" in current_explanation.lower() or "frac" in expression:
                templates = self.explanation_templates["differentiate"]["quotient_rule"]
                step["explanation"] = self._select_template(templates)
            
            elif "chain" in current_explanation.lower() or "composite" in current_explanation.lower():
                templates = self.explanation_templates["differentiate"]["chain_rule"]
                step["explanation"] = self._select_template(templates)
            
            elif i == len(steps) - 1:  # Last step (final result)
                templates = self.explanation_templates["differentiate"]["final"]
                
                if order > 1:
                    detail = f"This is the {self._ordinal(order)} derivative of the original function."
                else:
                    detail = "This is the derivative of the original function."
                
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(detail, context)
        
        return enhanced_steps
    
    def _explain_integration_steps(self, 
                                 steps: List[Dict[str, str]], 
                                 context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate explanations for integration steps.
        
        Args:
            steps: List of integration steps
            context: Additional context
            
        Returns:
            List of steps with enhanced explanations
        """
        # Make a copy of the steps to avoid modifying the original
        enhanced_steps = steps.copy()
        
        # Extract variable information for better explanations
        variable = context.get("variable")
        if isinstance(variable, str):
            var_name = variable
        elif isinstance(variable, sp.Symbol):
            var_name = variable.name
        else:
            var_name = "x"  # Default variable name
        
        # Check if this is a definite integral
        is_definite = False
        if "lower_bound" in context and "upper_bound" in context and context["lower_bound"] is not None and context["upper_bound"] is not None:
            is_definite = True
            lower_bound = context["lower_bound"]
            upper_bound = context["upper_bound"]
        
        # Enhance explanations based on step content
        for i, step in enumerate(enhanced_steps):
            current_explanation = step["explanation"]
            expression = step["expression"]
            
            # Check for different types of steps and enhance explanations
            if i == 0:  # First step (original integral)
                templates = self.explanation_templates["integrate"]["initial"]
                
                if is_definite:
                    detail = f"We need to evaluate the definite integral from {lower_bound} to {upper_bound}."
                else:
                    detail = f"We need to find the indefinite integral with respect to {var_name}."
                
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(detail, context)
            
            elif "power rule" in current_explanation.lower() or "x^n" in expression:
                templates = self.explanation_templates["integrate"]["power_rule"]
                step["explanation"] = self._select_template(templates)
            
            elif "substitution" in current_explanation.lower() or "u-substitution" in current_explanation.lower():
                templates = self.explanation_templates["integrate"]["substitution"]
                step["explanation"] = self._select_template(templates)
            
            elif "parts" in current_explanation.lower():
                templates = self.explanation_templates["integrate"]["parts"]
                step["explanation"] = self._select_template(templates)
            
            elif is_definite and ("evaluate" in current_explanation.lower() or "bound" in current_explanation.lower()):
                templates = self.explanation_templates["integrate"]["definite"]
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(
                    f"We substitute {var_name} = {upper_bound} and {var_name} = {lower_bound} into the antiderivative and subtract.", context
                )
            
            elif i == len(steps) - 1:  # Last step (final result)
                templates = self.explanation_templates["integrate"]["final"]
                
                if is_definite:
                    detail = "This is the value of the definite integral."
                else:
                    detail = "This is the indefinite integral of the original function, where C is an arbitrary constant."
                
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(detail, context)
        
        return enhanced_steps
    
    def _explain_limit_steps(self, 
                           steps: List[Dict[str, str]], 
                           context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate explanations for limit computation steps.
        
        Args:
            steps: List of limit computation steps
            context: Additional context
            
        Returns:
            List of steps with enhanced explanations
        """
        # Make a copy of the steps to avoid modifying the original
        enhanced_steps = steps.copy()
        
        # Extract limit information for better explanations
        variable = context.get("variable")
        if isinstance(variable, str):
            var_name = variable
        elif isinstance(variable, sp.Symbol):
            var_name = variable.name
        else:
            var_name = "x"  # Default variable name
        
        point = context.get("point")
        direction = context.get("direction", "both")
        
        # Direction text
        dir_text = ""
        if direction == "+":
            dir_text = "from the right"
        elif direction == "-":
            dir_text = "from the left"
        
        # Enhance explanations based on step content
        for i, step in enumerate(enhanced_steps):
            current_explanation = step["explanation"]
            expression = step["expression"]
            
            # Check for different types of steps and enhance explanations
            if i == 0:  # First step (original limit)
                templates = self.explanation_templates["limit"]["initial"]
                
                detail = f"We need to find what happens to the expression as {var_name} approaches {point} {dir_text}."
                
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(detail, context)
            
            elif "direct substitution" in current_explanation.lower():
                templates = self.explanation_templates["limit"]["direct_substitution"]
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(
                    f"We can simply substitute {var_name} = {point} into the expression.", context
                )
            
            elif "indeterminate" in current_explanation.lower():
                templates = self.explanation_templates["limit"]["indeterminate"]
                step["explanation"] = self._select_template(templates) + " " + self._customize_explanation(
                    "We need to use algebraic manipulation or special limit theorems.", context
                )
            
            elif "factor" in current_explanation.lower():
                templates = self.explanation_templates["limit"]["factoring"]
                step["explanation"] = self._select_template(templates)
            
            elif "l'hôpital" in current_explanation.lower() or "l'hopital" in current_explanation.lower():
                templates = self.explanation_templates["limit"]["lhopital"]
                step["explanation"] = self._select_template(templates)
            
            elif i == len(steps) - 1:  # Last step (final result)
                templates = self.explanation_templates["limit"]["final"]
                step["explanation"] = self._select_template(templates)
        
        return enhanced_steps
    
    def _explain_factoring_steps(self, 
                               steps: List[Dict[str, str]], 
                               context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate explanations for factoring steps.
        
        Args:
            steps: List of factoring steps
            context: Additional context
            
        Returns:
            List of steps with enhanced explanations
        """
        # For this implementation, we'll just return the original steps
        # A full implementation would enhance these explanations
        return steps
    
    def _explain_expansion_steps(self, 
                               steps: List[Dict[str, str]], 
                               context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate explanations for expansion steps.
        
        Args:
            steps: List of expansion steps
            context: Additional context
            
        Returns:
            List of steps with enhanced explanations
        """
        # For this implementation, we'll just return the original steps
        # A full implementation would enhance these explanations
        return steps
    
    def _explain_simplification_steps(self, 
                                    steps: List[Dict[str, str]], 
                                    context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate explanations for simplification steps.
        
        Args:
            steps: List of simplification steps
            context: Additional context
            
        Returns:
            List of steps with enhanced explanations
        """
        # For this implementation, we'll just return the original steps
        # A full implementation would enhance these explanations
        return steps
    
    def _select_template(self, templates: List[str]) -> str:
        """
        Select a template randomly from the available options.
        
        Args:
            templates: List of template strings
            
        Returns:
            Selected template string
        """
        import random
        return random.choice(templates)
    
    def _customize_explanation(self, 
                             detail: str, 
                             context: Dict[str, Any]) -> str:
        """
        Customize explanation with additional details.
        
        Args:
            detail: Additional detail to add
            context: Context information
            
        Returns:
            Customized detail string
        """
        # Add any context-specific customization here
        return detail
    
    def _ordinal(self, n: int) -> str:
        """
        Convert a number to its ordinal form (1st, 2nd, 3rd, etc.).
        
        Args:
            n: Number to convert
            
        Returns:
            Ordinal form of the number
        """
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

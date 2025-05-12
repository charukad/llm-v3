cat > math_processing/knowledge/knowledge_base.py << 'EOF'
"""
Mathematical Knowledge Base

This module provides a structured repository of mathematical knowledge, including concepts,
theorems, formulas, and relationships. It supports the step-by-step solution generation
by providing contextual information and references to mathematical principles.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import sympy as sp
from math_processing.knowledge.repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class MathematicalConcept:
    """Represents a mathematical concept with associated information."""
    
    def __init__(
        self,
        concept_id: str,
        name: str,
        domain: str,
        description: str,
        latex_representation: Optional[str] = None,
        symbolic_representation: Optional[sp.Expr] = None,
        related_concepts: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        properties: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a mathematical concept.
        
        Args:
            concept_id: Unique identifier for the concept
            name: Name of the concept
            domain: Mathematical domain (e.g., algebra, calculus)
            description: Textual description of the concept
            latex_representation: LaTeX representation if applicable
            symbolic_representation: SymPy representation if applicable
            related_concepts: List of related concept IDs
            examples: List of examples illustrating the concept
            properties: List of properties associated with the concept
        """
        self.concept_id = concept_id
        self.name = name
        self.domain = domain
        self.description = description
        self.latex_representation = latex_representation
        self.symbolic_representation = symbolic_representation
        self.related_concepts = related_concepts or []
        self.examples = examples or []
        self.properties = properties or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the concept to a dictionary representation."""
        result = {
            "concept_id": self.concept_id,
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "related_concepts": self.related_concepts,
            "examples": self.examples,
            "properties": self.properties
        }
        
        if self.latex_representation:
            result["latex_representation"] = self.latex_representation
            
        if self.symbolic_representation:
            # Convert symbolic representation to string for storage
            result["symbolic_representation"] = str(self.symbolic_representation)
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MathematicalConcept':
        """Create a concept from a dictionary representation."""
        # Handle symbolic representation if present
        symbolic_repr = None
        if "symbolic_representation" in data:
            try:
                symbolic_repr = sp.sympify(data["symbolic_representation"])
            except Exception as e:
                logger.warning(f"Could not parse symbolic representation for {data.get('name', 'unknown')}: {e}")
        
        return cls(
            concept_id=data["concept_id"],
            name=data["name"],
            domain=data["domain"],
            description=data["description"],
            latex_representation=data.get("latex_representation"),
            symbolic_representation=symbolic_repr,
            related_concepts=data.get("related_concepts", []),
            examples=data.get("examples", []),
            properties=data.get("properties", [])
        )


class MathematicalTheorem:
    """Represents a mathematical theorem or formula."""
    
    def __init__(
        self,
        theorem_id: str,
        name: str,
        domain: str,
        statement: str,
        latex_representation: str,
        symbolic_representation: Optional[sp.Expr] = None,
        proof: Optional[str] = None,
        related_concepts: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        conditions: Optional[str] = None
    ):
        """
        Initialize a mathematical theorem.
        
        Args:
            theorem_id: Unique identifier for the theorem
            name: Name of the theorem
            domain: Mathematical domain (e.g., algebra, calculus)
            statement: Textual statement of the theorem
            latex_representation: LaTeX representation of the theorem
            symbolic_representation: SymPy representation if applicable
            proof: Proof of the theorem
            related_concepts: List of related concept IDs
            examples: List of examples illustrating the theorem
            conditions: Conditions for the theorem's applicability
        """
        self.theorem_id = theorem_id
        self.name = name
        self.domain = domain
        self.statement = statement
        self.latex_representation = latex_representation
        self.symbolic_representation = symbolic_representation
        self.proof = proof
        self.related_concepts = related_concepts or []
        self.examples = examples or []
        self.conditions = conditions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the theorem to a dictionary representation."""
        result = {
            "theorem_id": self.theorem_id,
            "name": self.name,
            "domain": self.domain,
            "statement": self.statement,
            "latex_representation": self.latex_representation,
            "related_concepts": self.related_concepts,
            "examples": self.examples
        }
        
        if self.symbolic_representation:
            # Convert symbolic representation to string for storage
            result["symbolic_representation"] = str(self.symbolic_representation)
            
        if self.proof:
            result["proof"] = self.proof
            
        if self.conditions:
            result["conditions"] = self.conditions
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MathematicalTheorem':
        """Create a theorem from a dictionary representation."""
        # Handle symbolic representation if present
        symbolic_repr = None
        if "symbolic_representation" in data:
            try:
                symbolic_repr = sp.sympify(data["symbolic_representation"])
            except Exception as e:
                logger.warning(f"Could not parse symbolic representation for {data.get('name', 'unknown')}: {e}")
        
        return cls(
            theorem_id=data["theorem_id"],
            name=data["name"],
            domain=data["domain"],
            statement=data["statement"],
            latex_representation=data["latex_representation"],
            symbolic_representation=symbolic_repr,
            proof=data.get("proof"),
            related_concepts=data.get("related_concepts", []),
            examples=data.get("examples", []),
            conditions=data.get("conditions")
        )


class MathematicalFormula:
    """Represents a mathematical formula with associated context."""
    
    def __init__(
        self,
        formula_id: str,
        name: str,
        domain: str,
        description: str,
        latex_representation: str,
        symbolic_representation: Optional[sp.Expr] = None,
        variables: Optional[Dict[str, str]] = None,
        related_concepts: Optional[List[str]] = None,
        usage_examples: Optional[List[Dict[str, Any]]] = None,
        conditions: Optional[str] = None
    ):
        """
        Initialize a mathematical formula.
        
        Args:
            formula_id: Unique identifier for the formula
            name: Name of the formula
            domain: Mathematical domain (e.g., algebra, calculus)
            description: Textual description of the formula
            latex_representation: LaTeX representation of the formula
            symbolic_representation: SymPy representation if applicable
            variables: Dictionary mapping variable names to descriptions
            related_concepts: List of related concept IDs
            usage_examples: List of examples showing formula application
            conditions: Conditions for the formula's applicability
        """
        self.formula_id = formula_id
        self.name = name
        self.domain = domain
        self.description = description
        self.latex_representation = latex_representation
        self.symbolic_representation = symbolic_representation
        self.variables = variables or {}
        self.related_concepts = related_concepts or []
        self.usage_examples = usage_examples or []
        self.conditions = conditions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the formula to a dictionary representation."""
        result = {
            "formula_id": self.formula_id,
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "latex_representation": self.latex_representation,
            "variables": self.variables,
            "related_concepts": self.related_concepts,
            "usage_examples": self.usage_examples
        }
        
        if self.symbolic_representation:
            # Convert symbolic representation to string for storage
            result["symbolic_representation"] = str(self.symbolic_representation)
            
        if self.conditions:
            result["conditions"] = self.conditions
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MathematicalFormula':
        """Create a formula from a dictionary representation."""
        # Handle symbolic representation if present
        symbolic_repr = None
        if "symbolic_representation" in data:
            try:
                symbolic_repr = sp.sympify(data["symbolic_representation"])
            except Exception as e:
                logger.warning(f"Could not parse symbolic representation for {data.get('name', 'unknown')}: {e}")
        
        return cls(
            formula_id=data["formula_id"],
            name=data["name"],
            domain=data["domain"],
            description=data["description"],
            latex_representation=data["latex_representation"],
            symbolic_representation=symbolic_repr,
            variables=data.get("variables", {}),
            related_concepts=data.get("related_concepts", []),
            usage_examples=data.get("usage_examples", []),
            conditions=data.get("conditions")
        )


class MathKnowledgeBase:
    """
    Mathematical knowledge base for storing and retrieving mathematical knowledge.
    
    This class provides access to mathematical concepts, theorems, and formulas,
    supporting the solution generation process with contextual information.
    """
    
    def __init__(self, repository: Optional[KnowledgeRepository] = None):
        """
        Initialize the knowledge base.
        
        Args:
            repository: Repository for storing mathematical knowledge
        """
        self.repository = repository
        
        # In-memory cache for frequently accessed items
        self._concept_cache = {}
        self._theorem_cache = {}
        self._formula_cache = {}
        
        # Initialize with core mathematical knowledge if repository is not provided
        if not repository:
            self._initialize_core_knowledge()
    
    def _initialize_core_knowledge(self):
        """Initialize the knowledge base with core mathematical knowledge."""
        # Add basic concepts
        self._add_core_concepts()
        
        # Add basic theorems
        self._add_core_theorems()
        
        # Add basic formulas
        self._add_core_formulas()
    
    def _add_core_concepts(self):
        """Add core mathematical concepts to the knowledge base."""
        # Algebra concepts
        algebra_concepts = [
            MathematicalConcept(
                concept_id="polynomial",
                name="Polynomial",
                domain="algebra",
                description="An expression consisting of variables and coefficients using only addition, subtraction, multiplication, and non-negative integer exponents.",
                latex_representation="a_nx^n + a_{n-1}x^{n-1} + \\ldots + a_1x + a_0",
                related_concepts=["equation", "function", "factorization"]
            ),
            MathematicalConcept(
                concept_id="quadratic_equation",
                name="Quadratic Equation",
                domain="algebra",
                description="A polynomial equation of the second degree, in the form ax² + bx + c = 0, where a ≠ 0.",
                latex_representation="ax^2 + bx + c = 0",
                related_concepts=["polynomial", "equation", "quadratic_formula"],
                examples=[
                    {"expression": "3x^2 - 5x + 2 = 0", "description": "A quadratic equation with a=3, b=-5, c=2"}
                ],
                properties=[
                    {"name": "Discriminant", "description": "b² - 4ac determines the number and nature of solutions"}
                ]
            )
        ]
        
        # Calculus concepts
        calculus_concepts = [
            MathematicalConcept(
                concept_id="derivative",
                name="Derivative",
                domain="calculus",
                description="A measure of how a function changes as its input changes.",
                latex_representation="\\frac{d}{dx}f(x) = \\lim_{h \\to 0}\\frac{f(x+h) - f(x)}{h}",
                related_concepts=["function", "limit", "differentiation_rules"]
            ),
            MathematicalConcept(
                concept_id="integral",
                name="Integral",
                domain="calculus",
                description="A generalization of addition, multiplication, and area calculation.",
                latex_representation="\\int f(x) dx",
                related_concepts=["function", "antiderivative", "fundamental_theorem_calculus"]
            )
        ]
        
        # Linear algebra concepts
        linear_algebra_concepts = [
            MathematicalConcept(
                concept_id="matrix",
                name="Matrix",
                domain="linear_algebra",
                description="A rectangular array of numbers, symbols, or expressions arranged in rows and columns.",
                latex_representation="\\begin{bmatrix} a_{11} & a_{12} & \\cdots & a_{1n} \\\\ a_{21} & a_{22} & \\cdots & a_{2n} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ a_{m1} & a_{m2} & \\cdots & a_{mn} \\end{bmatrix}",
                related_concepts=["determinant", "vector", "linear_system"]
            ),
            MathematicalConcept(
                concept_id="determinant",
                name="Determinant",
                domain="linear_algebra",
                description="A scalar value that can be computed from the elements of a square matrix and encodes certain properties of the linear transformation described by the matrix.",
                latex_representation="\\det(A) = |A|",
                related_concepts=["matrix", "inverse_matrix", "eigenvalue"]
            )
        ]
        
        # Add all concepts to the cache
        all_concepts = algebra_concepts + calculus_concepts + linear_algebra_concepts
        for concept in all_concepts:
            self._concept_cache[concept.concept_id] = concept
    
    def _add_core_theorems(self):

        """Add core mathematical theorems to the knowledge base."""
        # Algebra theorems
        algebra_theorems = [
            MathematicalTheorem(
                theorem_id="quadratic_formula",
                name="Quadratic Formula",
                domain="algebra",
                statement="For a quadratic equation ax² + bx + c = 0, the solutions are x = (-b ± √(b² - 4ac))/(2a).",
                latex_representation="x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
                symbolic_representation=sp.solve(sp.Eq(sp.Symbol('a')*sp.Symbol('x')**2 + sp.Symbol('b')*sp.Symbol('x') + sp.Symbol('c'), 0), sp.Symbol('x')),
                related_concepts=["quadratic_equation", "polynomial"],
                examples=[
                    {"equation": "2x^2 - 5x + 2 = 0", "solutions": "x = 2 or x = 1/2"}
                ],
                conditions="Valid for any quadratic equation where a ≠ 0"
            ),
            MathematicalTheorem(
                theorem_id="factor_theorem",
                name="Factor Theorem",
                domain="algebra",
                statement="A polynomial P(x) has a factor (x - a) if and only if P(a) = 0.",
                latex_representation="(x - a) \\text{ is a factor of } P(x) \\iff P(a) = 0",
                related_concepts=["polynomial", "factorization", "remainder_theorem"],
                examples=[
                    {"polynomial": "x^3 - 4x^2 + 5x - 2", "value": "x = 1", "evaluation": "P(1) = 0, so (x - 1) is a factor"}
                ]
            )
        ]
        
        # Calculus theorems
        calculus_theorems = [
            MathematicalTheorem(
                theorem_id="fundamental_theorem_calculus",
                name="Fundamental Theorem of Calculus",
                domain="calculus",
                statement="If f is continuous on [a, b] and F is an antiderivative of f, then the definite integral of f from a to b equals F(b) - F(a).",
                latex_representation="\\int_{a}^{b} f(x) dx = F(b) - F(a), \\text{ where } F'(x) = f(x)",
                related_concepts=["integral", "derivative", "antiderivative"],
                examples=[
                    {"function": "f(x) = x^2", "interval": "[1, 3]", "calculation": "∫₁³ x² dx = [x³/3]₁³ = 3³/3 - 1³/3 = 9 - 1/3 = 8⅔"}
                ]
            ),
            MathematicalTheorem(
                theorem_id="product_rule",
                name="Product Rule",
                domain="calculus",
                statement="The derivative of a product of two functions is the first function times the derivative of the second, plus the second function times the derivative of the first.",
                latex_representation="\\frac{d}{dx}[f(x)g(x)] = f(x)\\frac{d}{dx}[g(x)] + g(x)\\frac{d}{dx}[f(x)]",
                related_concepts=["derivative", "chain_rule", "quotient_rule"],
                examples=[
                    {"function": "h(x) = x² · sin(x)", "derivative": "h'(x) = x² · cos(x) + 2x · sin(x)"}
                ]
            ),
            MathematicalTheorem(
                theorem_id="chain_rule",
                name="Chain Rule",
                domain="calculus",
                statement="If y = f(g(x)), then dy/dx = f'(g(x)) · g'(x).",
                latex_representation="\\frac{d}{dx}[f(g(x))] = f'(g(x)) \\cdot g'(x)",
                related_concepts=["derivative", "product_rule", "composite_function"],
                examples=[
                    {"function": "h(x) = sin(x²)", "derivative": "h'(x) = cos(x²) · 2x"}
                ]
            )
        ]
        
        # Linear algebra theorems
        linear_algebra_theorems = [
            MathematicalTheorem(
                theorem_id="determinant_product",
                name="Determinant of Product",
                domain="linear_algebra",
                statement="The determinant of a product of matrices equals the product of their determinants.",
                latex_representation="\\det(AB) = \\det(A) \\cdot \\det(B)",
                related_concepts=["determinant", "matrix", "matrix_multiplication"],
                examples=[
                    {"matrices": "A = [[1, 2], [3, 4]], B = [[2, 0], [1, 3]]", "calculation": "det(AB) = det(A) · det(B) = -2 · 6 = -12"}
                ]
            ),
            MathematicalTheorem(
                theorem_id="eigenvector_definition",
                name="Eigenvector and Eigenvalue Relation",
                domain="linear_algebra",
                statement="A non-zero vector v is an eigenvector of a square matrix A if there exists a scalar λ (the eigenvalue) such that Av = λv.",
                latex_representation="A\\vec{v} = \\lambda\\vec{v}",
                related_concepts=["eigenvalue", "matrix", "linear_transformation"],
                examples=[
                    {"matrix": "A = [[3, 1], [2, 2]]", "eigenpair": "λ = 4, v = [1, 1]", "verification": "A·v = [4, 4] = 4·[1, 1]"}
                ]
            )
        ]
        
        # Add all theorems to the cache
        all_theorems = algebra_theorems + calculus_theorems + linear_algebra_theorems
        for theorem in all_theorems:
            self._theorem_cache[theorem.theorem_id] = theorem
    
    def _add_core_formulas(self):
        """Add core mathematical formulas to the knowledge base."""
        # Algebra formulas
        algebra_formulas = [
            MathematicalFormula(
                formula_id="binomial_expansion",
                name="Binomial Expansion",
                domain="algebra",
                description="The expansion of (a + b)^n as a sum of terms of the form (n choose k) a^(n-k) b^k.",
                latex_representation="(a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k",
                variables={
                    "a": "First term in the binomial",
                    "b": "Second term in the binomial",
                    "n": "Power to which the binomial is raised"
                },
                related_concepts=["binomial_coefficient", "polynomial", "pascal_triangle"],
                usage_examples=[
                    {"expression": "(x + y)³", "expansion": "x³ + 3x²y + 3xy² + y³"}
                ]
            ),
            MathematicalFormula(
                formula_id="difference_of_squares",
                name="Difference of Squares",
                domain="algebra",
                description="The factorization of a² - b² as (a + b)(a - b).",
                latex_representation="a^2 - b^2 = (a + b)(a - b)",
                variables={
                    "a": "First term",
                    "b": "Second term"
                },
                related_concepts=["factorization", "polynomial"],
                usage_examples=[
                    {"expression": "x² - 4", "factorization": "(x + 2)(x - 2)"}
                ]
            )
        ]
        
        # Calculus formulas
        calculus_formulas = [
            MathematicalFormula(
                formula_id="power_rule_integration",
                name="Power Rule for Integration",
                domain="calculus",
                description="The integral of x^n with respect to x is x^(n+1)/(n+1) + C, where n ≠ -1.",
                latex_representation="\\int x^n dx = \\frac{x^{n+1}}{n+1} + C, \\quad n \\neq -1",
                variables={
                    "x": "The variable of integration",
                    "n": "The power to which x is raised",
                    "C": "Constant of integration"
                },
                related_concepts=["integral", "power_rule_differentiation"],
                usage_examples=[
                    {"expression": "∫ x³ dx", "result": "x⁴/4 + C"}
                ],
                conditions="Valid for n ≠ -1"
            ),
            MathematicalFormula(
                formula_id="power_rule_differentiation",
                name="Power Rule for Differentiation",
                domain="calculus",
                description="The derivative of x^n with respect to x is n·x^(n-1).",
                latex_representation="\\frac{d}{dx}[x^n] = nx^{n-1}",
                variables={
                    "x": "The variable of differentiation",
                    "n": "The power to which x is raised"
                },
                related_concepts=["derivative", "power_rule_integration"],
                usage_examples=[
                    {"expression": "d/dx[x⁵]", "result": "5x⁴"}
                ]
            )
        ]
        
        # Linear algebra formulas
        linear_algebra_formulas = [
            MathematicalFormula(
                formula_id="determinant_2x2",
                name="Determinant of 2×2 Matrix",
                domain="linear_algebra",
                description="The determinant of a 2×2 matrix is calculated as ad - bc.",
                latex_representation="\\det\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} = ad - bc",
                variables={
                    "a": "Element at position (1,1)",
                    "b": "Element at position (1,2)",
                    "c": "Element at position (2,1)",
                    "d": "Element at position (2,2)"
                },
                related_concepts=["determinant", "matrix"],
                usage_examples=[
                    {"matrix": "[[3, 2], [1, 4]]", "calculation": "det = 3·4 - 2·1 = 12 - 2 = 10"}
                ]
            ),
            MathematicalFormula(
                formula_id="cramer_rule",
                name="Cramer's Rule",
                domain="linear_algebra",
                description="A method for solving systems of linear equations using determinants.",
                latex_representation="x_i = \\frac{\\det(A_i)}{\\det(A)}",
                variables={
                    "A": "Coefficient matrix",
                    "A_i": "Matrix formed by replacing the i-th column of A with the constants vector"
                },
                related_concepts=["determinant", "linear_system", "matrix"],
                usage_examples=[
                    {"system": "3x + 2y = 7, x + 4y = 8", "solution": "x = 1, y = 1.75"}
                ],
                conditions="Valid when det(A) ≠ 0"
            )
        ]
        
        # Add all formulas to the cache
        all_formulas = algebra_formulas + calculus_formulas + linear_algebra_formulas
        for formula in all_formulas:
            self._formula_cache[formula.formula_id] = formula
    
    def get_concept(self, concept_id: str) -> Optional[MathematicalConcept]:
        """
        Get a mathematical concept by its ID.
        
        Args:
            concept_id: Identifier for the concept
            
        Returns:
            The mathematical concept or None if not found
        """
        # First check the cache
        if concept_id in self._concept_cache:
            return self._concept_cache[concept_id]
        
        # If not in cache and we have a repository, try to fetch from it
        if self.repository:
            concept_data = self.repository.get_concept(concept_id)
            if concept_data:
                concept = MathematicalConcept.from_dict(concept_data)
                self._concept_cache[concept_id] = concept
                return concept
        
        return None
    
    def get_theorem(self, theorem_id: str) -> Optional[MathematicalTheorem]:
        """
        Get a mathematical theorem by its ID.
        
        Args:
            theorem_id: Identifier for the theorem
            
        Returns:
            The mathematical theorem or None if not found
        """
        # First check the cache
        if theorem_id in self._theorem_cache:
            return self._theorem_cache[theorem_id]
        
        # If not in cache and we have a repository, try to fetch from it
        if self.repository:
            theorem_data = self.repository.get_theorem(theorem_id)
            if theorem_data:
                theorem = MathematicalTheorem.from_dict(theorem_data)
                self._theorem_cache[theorem_id] = theorem
                return theorem
        
        return None
    
    def get_formula(self, formula_id: str) -> Optional[MathematicalFormula]:
        """
        Get a mathematical formula by its ID.
        
        Args:
            formula_id: Identifier for the formula
            
        Returns:
            The mathematical formula or None if not found
        """
        # First check the cache
        if formula_id in self._formula_cache:
            return self._formula_cache[formula_id]
        
        # If not in cache and we have a repository, try to fetch from it
        if self.repository:
            formula_data = self.repository.get_formula(formula_id)
            if formula_data:
                formula = MathematicalFormula.from_dict(formula_data)
                self._formula_cache[formula_id] = formula
                return formula
        
        return None
    
    def search_concepts_by_domain(self, domain: str) -> List[MathematicalConcept]:
        """
        Search for concepts in a specific domain.
        
        Args:
            domain: Mathematical domain to search in
            
        Returns:
            List of concepts in the specified domain
        """
        result = []
        
        # First check the cache
        for concept in self._concept_cache.values():
            if concept.domain == domain:
                result.append(concept)
        
        # If we have a repository, search in it as well
        if self.repository:
            concept_data_list = self.repository.search_concepts({"domain": domain})
            for concept_data in concept_data_list:
                # Skip concepts we already have in the result
                if concept_data["concept_id"] in [c.concept_id for c in result]:
                    continue
                
                concept = MathematicalConcept.from_dict(concept_data)
                self._concept_cache[concept.concept_id] = concept
                result.append(concept)
        
        return result
    
    def search_theorems_by_domain(self, domain: str) -> List[MathematicalTheorem]:
        """
        Search for theorems in a specific domain.
        
        Args:
            domain: Mathematical domain to search in
            
        Returns:
            List of theorems in the specified domain
        """
        result = []
        
        # First check the cache
        for theorem in self._theorem_cache.values():
            if theorem.domain == domain:
                result.append(theorem)
        
        # If we have a repository, search in it as well
        if self.repository:
            theorem_data_list = self.repository.search_theorems({"domain": domain})
            for theorem_data in theorem_data_list:
                # Skip theorems we already have in the result
                if theorem_data["theorem_id"] in [t.theorem_id for t in result]:
                    continue
                
                theorem = MathematicalTheorem.from_dict(theorem_data)
                self._theorem_cache[theorem.theorem_id] = theorem
                result.append(theorem)
        
        return result
    
    def search_formulas_by_domain(self, domain: str) -> List[MathematicalFormula]:
        """
        Search for formulas in a specific domain.
        
        Args:
            domain: Mathematical domain to search in
            
        Returns:
            List of formulas in the specified domain
        """
        result = []
        
        # First check the cache
        for formula in self._formula_cache.values():
            if formula.domain == domain:
                result.append(formula)
        
        # If we have a repository, search in it as well
        if self.repository:
            formula_data_list = self.repository.search_formulas({"domain": domain})
            for formula_data in formula_data_list:
                # Skip formulas we already have in the result
                if formula_data["formula_id"] in [f.formula_id for f in result]:
                    continue
                
                formula = MathematicalFormula.from_dict(formula_data)
                self._formula_cache[formula.formula_id] = formula
                result.append(formula)
        
        return result
    
    def search_by_name(self, name: str) -> Dict[str, List]:
        """
        Search for mathematical knowledge by name.
        
        Args:
            name: Name or partial name to search for
            
        Returns:
            Dictionary with concepts, theorems, and formulas that match the search
        """
        result = {
            "concepts": [],
            "theorems": [],
            "formulas": []
        }
        
        # First check the cache
        name_lower = name.lower()
        
        for concept in self._concept_cache.values():
            if name_lower in concept.name.lower():
                result["concepts"].append(concept)
        
        for theorem in self._theorem_cache.values():
            if name_lower in theorem.name.lower():
                result["theorems"].append(theorem)
        
        for formula in self._formula_cache.values():
            if name_lower in formula.name.lower():
                result["formulas"].append(formula)
        
        # If we have a repository, search in it as well
        if self.repository:
            # This is a simplified implementation - in practice, we'd need a more sophisticated search
            search_criteria = {"name_contains": name}
            
            concept_data_list = self.repository.search_concepts(search_criteria)
            for concept_data in concept_data_list:
                # Skip concepts we already have in the result
                if concept_data["concept_id"] in [c.concept_id for c in result["concepts"]]:
                    continue
                
                concept = MathematicalConcept.from_dict(concept_data)
                self._concept_cache[concept.concept_id] = concept
                result["concepts"].append(concept)
            
            theorem_data_list = self.repository.search_theorems(search_criteria)
            for theorem_data in theorem_data_list:
                # Skip theorems we already have in the result
                if theorem_data["theorem_id"] in [t.theorem_id for t in result["theorems"]]:
                    continue
                
                theorem = MathematicalTheorem.from_dict(theorem_data)
                self._theorem_cache[theorem.theorem_id] = theorem
                result["theorems"].append(theorem)
            
            formula_data_list = self.repository.search_formulas(search_criteria)
            for formula_data in formula_data_list:
                # Skip formulas we already have in the result
                if formula_data["formula_id"] in [f.formula_id for f in result["formulas"]]:
                    continue
                
                formula = MathematicalFormula.from_dict(formula_data)
                self._formula_cache[formula.formula_id] = formula
                result["formulas"].append(formula)
        
        return result
    
    def get_related_concepts(self, concept_id: str, depth: int = 1) -> List[MathematicalConcept]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept_id: ID of the concept to get related concepts for
            depth: Depth of relationship traversal (default is 1, direct relationships only)
            
        Returns:
            List of related concepts
        """
        result = []
        visited_ids = set()
        
        # Get the initial concept
        concept = self.get_concept(concept_id)
        if not concept:
            return result
        
        # Add the concept itself to the visited set
        visited_ids.add(concept_id)
        
        # Process the related concepts
        related_ids = concept.related_concepts
        for related_id in related_ids:
            if related_id not in visited_ids:
                related_concept = self.get_concept(related_id)
                if related_concept:
                    result.append(related_concept)
                    visited_ids.add(related_id)
        
        # If depth is greater than 1, recursively get related concepts
        if depth > 1:
            for related_id in related_ids:
                next_level_related = self.get_related_concepts(related_id, depth - 1)
                for next_concept in next_level_related:
                    if next_concept.concept_id not in visited_ids:
                        result.append(next_concept)
                        visited_ids.add(next_concept.concept_id)
        
        return result
    
    def get_applicable_theorems(self, domain: str, expression_type: str) -> List[MathematicalTheorem]:
        """
        Get theorems applicable to a specific type of mathematical expression.
        
        Args:
            domain: Mathematical domain (e.g., algebra, calculus)
            expression_type: Type of expression (e.g., quadratic_equation, derivative)
            
        Returns:
            List of applicable theorems
        """
        result = []
        
        # First get all theorems in the domain
        domain_theorems = self.search_theorems_by_domain(domain)
        
        # Filter based on expression type - this is a simple implementation
        # In practice, we'd need more sophisticated matching based on the properties
        # of the expression
        for theorem in domain_theorems:
            # Check if any related concept matches the expression type
            if expression_type in theorem.related_concepts:
                result.append(theorem)
        
        return result
    
    def get_applicable_formulas(self, domain: str, expression_type: str) -> List[MathematicalFormula]:
        """
        Get formulas applicable to a specific type of mathematical expression.
        
        Args:
            domain: Mathematical domain (e.g., algebra, calculus)
            expression_type: Type of expression (e.g., quadratic_equation, derivative)
            
        Returns:
            List of applicable formulas
        """
        result = []
        
        # First get all formulas in the domain
        domain_formulas = self.search_formulas_by_domain(domain)
        
        # Filter based on expression type
        for formula in domain_formulas:
            # Check if any related concept matches the expression type
            if expression_type in formula.related_concepts:
                result.append(formula)
        
        return result
    
    def get_context_for_solution(self, domain: str, operation: str, expression_type: str) -> Dict[str, Any]:
        """
        Get contextual information for a solution step.
        
        Args:
            domain: Mathematical domain (e.g., algebra, calculus)
            operation: Operation being performed (e.g., solve, differentiate)
            expression_type: Type of expression being operated on
            
        Returns:
            Dictionary with relevant concepts, theorems, and formulas
        """
        context = {
            "concepts": [],
            "theorems": [],
            "formulas": []
        }
        
        # Get concepts related to the domain and operation
        if domain == "algebra" and operation == "solve":
            # For algebra solving, include concepts like equations, polynomials
            for concept_id in ["polynomial", "quadratic_equation"]:
                concept = self.get_concept(concept_id)
                if concept:
                    context["concepts"].append(concept)
        
        elif domain == "calculus" and operation == "differentiate":
            # For differentiation, include concepts like derivatives, rules
            for concept_id in ["derivative"]:
                concept = self.get_concept(concept_id)
                if concept:
                    context["concepts"].append(concept)
        
        elif domain == "calculus" and operation == "integrate":
            # For integration, include concepts like integrals, antiderivatives
            for concept_id in ["integral"]:
                concept = self.get_concept(concept_id)
                if concept:
                    context["concepts"].append(concept)
        
        # Get applicable theorems
        if domain == "algebra" and operation == "solve" and expression_type == "quadratic_equation":
            # For quadratic equations, include the quadratic formula
            theorem = self.get_theorem("quadratic_formula")
            if theorem:
                context["theorems"].append(theorem)
        
        elif domain == "calculus" and operation == "differentiate":
            # For differentiation, include relevant rules
            for theorem_id in ["product_rule", "chain_rule"]:
                theorem = self.get_theorem(theorem_id)
                if theorem:
                    context["theorems"].append(theorem)
        
        # Get applicable formulas
        if domain == "calculus" and operation == "differentiate":
            formula = self.get_formula("power_rule_differentiation")
            if formula:
                context["formulas"].append(formula)
        
        elif domain == "calculus" and operation == "integrate":
            formula = self.get_formula("power_rule_integration")
            if formula:
                context["formulas"].append(formula)
        
        return context
    
    def add_concept(self, concept: MathematicalConcept) -> bool:
        """
        Add a new concept to the knowledge base.
        
        Args:
            concept: Concept to add
            
        Returns:
            True if the concept was added successfully, False otherwise
        """
        # Add to the cache
        self._concept_cache[concept.concept_id] = concept
        
        # If we have a repository, add to it as well
        if self.repository:
            return self.repository.add_concept(concept.to_dict())
        
        return True
    
    def add_theorem(self, theorem: MathematicalTheorem) -> bool:
        """
        Add a new theorem to the knowledge base.
        
        Args:
            theorem: Theorem to add
            
        Returns:
            True if the theorem was added successfully, False otherwise
        """
        # Add to the cache
        self._theorem_cache[theorem.theorem_id] = theorem
        
        # If we have a repository, add to it as well
        if self.repository:
            return self.repository.add_theorem(theorem.to_dict())
        
        return True
    
    def add_formula(self, formula: MathematicalFormula) -> bool:
        """
        Add a new formula to the knowledge base.
        
        Args:
            formula: Formula to add
            
        Returns:
            True if the formula was added successfully, False otherwise
        """
        # Add to the cache
        self._formula_cache[formula.formula_id] = formula
        
        # If we have a repository, add to it as well
        if self.repository:
            return self.repository.add_formula(formula.to_dict())
        
        return True
EOF
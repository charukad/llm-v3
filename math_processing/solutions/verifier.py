"""
Solution Verification System

This module provides mechanisms for verifying the correctness of mathematical solutions
using multiple verification methods. It supports verification for various mathematical
domains and problem types.
"""

import sympy as sp
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import math

logger = logging.getLogger(__name__)

class VerificationResult:
    """Represents the result of a solution verification."""
    
    def __init__(
        self,
        is_correct: bool,
        verification_method: str,
        details: Dict[str, Any],
        confidence_score: float,
        error_message: Optional[str] = None
    ):
        """
        Initialize a verification result.
        
        Args:
            is_correct: Whether the solution is correct
            verification_method: Method used for verification
            details: Detailed information about the verification
            confidence_score: Confidence in the verification (0.0 to 1.0)
            error_message: Error message if verification failed
        """
        self.is_correct = is_correct
        self.verification_method = verification_method
        self.details = details
        self.confidence_score = max(0.0, min(1.0, confidence_score))  # Ensure range [0, 1]
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the verification result to a dictionary."""
        result = {
            "is_correct": self.is_correct,
            "verification_method": self.verification_method,
            "details": self.details,
            "confidence_score": self.confidence_score
        }
        
        if self.error_message:
            result["error_message"] = self.error_message
            
        return result


class SolutionVerifier:
    """
    Verifies the correctness of mathematical solutions.
    
    This class provides methods to verify solutions using multiple approaches,
    including symbolic verification, numerical evaluation, and domain-specific
    verification techniques.
    """
    
    def __init__(self):
        """Initialize the solution verifier."""
        self.tolerance = 1e-10  # Numerical comparison tolerance
    
    def verify_solution(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any],
        domain: str,
        verification_methods: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify a mathematical solution using appropriate methods.
        
        Args:
            problem: Problem description including type and parameters
            solution: Solution to verify
            domain: Mathematical domain (algebra, calculus, etc.)
            verification_methods: List of verification methods to use (if None, use all applicable)
            
        Returns:
            Verification result
        """
        # Determine the problem type
        problem_type = problem.get("problem_type", "")
        
        # Select verification methods if not specified
        if verification_methods is None:
            verification_methods = self._get_default_verification_methods(domain, problem_type)
        
        # Apply verification methods
        results = []
        for method in verification_methods:
            try:
                if method == "symbolic":
                    results.append(self._verify_symbolic(problem, solution, domain))
                elif method == "numerical":
                    results.append(self._verify_numerical(problem, solution, domain))
                elif method == "substitution":
                    results.append(self._verify_by_substitution(problem, solution, domain))
                elif method == "derivative":
                    results.append(self._verify_derivative(problem, solution))
                elif method == "integral":
                    results.append(self._verify_integral(problem, solution))
                elif method == "limit":
                    results.append(self._verify_limit(problem, solution))
                elif method == "matrix":
                    results.append(self._verify_matrix_operation(problem, solution))
                else:
                    logger.warning(f"Unknown verification method: {method}")
            except Exception as e:
                logger.error(f"Error in verification method {method}: {e}")
                results.append(VerificationResult(
                    is_correct=False,
                    verification_method=method,
                    details={"error": str(e)},
                    confidence_score=0.0,
                    error_message=f"Verification method {method} failed: {e}"
                ))
        
        # If no successful verification, return the first error
        if not results:
            return VerificationResult(
                is_correct=False,
                verification_method="none",
                details={"error": "No verification methods applied"},
                confidence_score=0.0,
                error_message="Verification failed: no methods were applied"
            )
        
        # Combine results - if any method fails, the solution is incorrect
        is_correct = all(result.is_correct for result in results)
        
        # Calculate average confidence score from successful verifications
        confidence_scores = [result.confidence_score for result in results if result.is_correct]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Collect details from all verifications
        combined_details = {}
        for i, result in enumerate(results):
            combined_details[f"method_{i+1}_{result.verification_method}"] = result.details
        
        # Determine error message if any
        error_messages = [result.error_message for result in results if result.error_message]
        error_message = "; ".join(error_messages) if error_messages else None
        
        # Combine into final result
        return VerificationResult(
            is_correct=is_correct,
            verification_method="multiple" if len(results) > 1 else results[0].verification_method,
            details=combined_details,
            confidence_score=avg_confidence,
            error_message=error_message
        )
    
    def _get_default_verification_methods(self, domain: str, problem_type: str) -> List[str]:
        """
        Get default verification methods for a domain and problem type.
        
        Args:
            domain: Mathematical domain
            problem_type: Type of problem
            
        Returns:
            List of verification methods
        """
        if domain == "algebra":
            if problem_type in ["equation", "quadratic_equation", "polynomial_equation"]:
                return ["symbolic", "substitution"]
            elif problem_type in ["expression", "simplification"]:
                return ["symbolic"]
            else:
                return ["symbolic", "numerical"]
        
        elif domain == "calculus":
            if problem_type == "derivative":
                return ["derivative", "symbolic"]
            elif problem_type == "integral":
                return ["integral", "symbolic"]
            elif problem_type == "limit":
                return ["limit", "symbolic"]
            else:
                return ["symbolic", "numerical"]
        
        elif domain == "linear_algebra":
            if problem_type in ["matrix_operation", "determinant", "eigenvalue"]:
                return ["matrix", "symbolic"]
            else:
                return ["symbolic", "numerical"]
        
        # Default fallback
        return ["symbolic", "numerical"]
    
    def _verify_symbolic(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any],
        domain: str
    ) -> VerificationResult:
        """
        Verify a solution using symbolic computation.
        
        Args:
            problem: Problem description
            solution: Solution to verify
            domain: Mathematical domain
            
        Returns:
            Verification result
        """
        problem_type = problem.get("problem_type", "")
        
        try:
            if domain == "algebra" and problem_type in ["equation", "quadratic_equation", "polynomial_equation"]:
                return self._verify_equation_solution(problem, solution)
            
            elif domain == "calculus" and problem_type == "derivative":
                return self._verify_derivative(problem, solution)
            
            elif domain == "calculus" and problem_type == "integral":
                return self._verify_integral(problem, solution)
            
            elif domain == "linear_algebra" and problem_type == "matrix_operation":
                return self._verify_matrix_operation(problem, solution)
            
            else:
                # Generic symbolic verification
                # Compare the solution expression with the expected result
                if "expression" in problem and "result" in solution:
                    problem_expr = self._parse_expression(problem["expression"])
                    solution_expr = self._parse_expression(solution["result"])
                    
                    # Check if the expressions are equivalent
                    difference = sp.simplify(problem_expr - solution_expr)
                    is_correct = difference == 0
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="symbolic",
                        details={
                            "problem_expression": str(problem_expr),
                            "solution_expression": str(solution_expr),
                            "difference": str(difference)
                        },
                        confidence_score=1.0 if is_correct else 0.0,
                        error_message=None if is_correct else "Solution does not match expected result"
                    )
                else:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="symbolic",
                        details={"error": "Missing required fields for symbolic verification"},
                        confidence_score=0.0,
                        error_message="Cannot verify: missing expression or result"
                    )
                
        except Exception as e:
            logger.error(f"Error in symbolic verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="symbolic",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Symbolic verification failed: {e}"
            )
    
    def _verify_numerical(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any],
        domain: str
    ) -> VerificationResult:
        """
        Verify a solution using numerical evaluation.
        
        Args:
            problem: Problem description
            solution: Solution to verify
            domain: Mathematical domain
            
        Returns:
            Verification result
        """
        problem_type = problem.get("problem_type", "")
        
        try:
            if "expression" in problem and "result" in solution:
                # Parse expressions
                problem_expr = self._parse_expression(problem["expression"])
                solution_expr = self._parse_expression(solution["result"])
                
                # Get variables from the expressions
                variables = problem_expr.free_symbols.union(solution_expr.free_symbols)
                
                # If no variables, can directly compare
                if not variables:
                    # Evaluate the expressions
                    problem_value = float(problem_expr.evalf())
                    solution_value = float(solution_expr.evalf())
                    
                    # Compare with tolerance
                    is_correct = abs(problem_value - solution_value) < self.tolerance
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="numerical",
                        details={
                            "problem_value": problem_value,
                            "solution_value": solution_value,
                            "difference": abs(problem_value - solution_value),
                            "tolerance": self.tolerance
                        },
                        confidence_score=1.0 if is_correct else 0.0,
                        error_message=None if is_correct else "Numerical values do not match within tolerance"
                    )
                
                # For expressions with variables, sample values and compare
                # This is a simplified approach - in practice, we'd need more
                # sophisticated sampling strategies
                else:
                    # Define test points
                    num_points = 5
                    test_points = []
                    for _ in range(num_points):
                        point = {}
                        for var in variables:
                            # Sample values in [-10, 10]
                            point[var] = np.random.uniform(-10, 10)
                        test_points.append(point)
                    
                    # Evaluate expressions at test points
                    matches = 0
                    mismatches = []
                    
                    for point in test_points:
                        try:
                            # Substitute values and evaluate
                            problem_value = float(problem_expr.subs(point).evalf())
                            solution_value = float(solution_expr.subs(point).evalf())
                            
                            # Check if values match within tolerance
                            if abs(problem_value - solution_value) < self.tolerance:
                                matches += 1
                            else:
                                mismatches.append({
                                    "point": {str(var): val for var, val in point.items()},
                                    "problem_value": problem_value,
                                    "solution_value": solution_value,
                                    "difference": abs(problem_value - solution_value)
                                })
                        except Exception as e:
                            # Skip points that cause evaluation errors
                            logger.warning(f"Error evaluating at point {point}: {e}")
                    
                    # Check if all evaluated points match
                    is_correct = matches == num_points
                    confidence_score = matches / num_points
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="numerical",
                        details={
                            "total_points": num_points,
                            "matching_points": matches,
                            "mismatches": mismatches
                        },
                        confidence_score=confidence_score,
                        error_message=None if is_correct else f"{num_points - matches} of {num_points} test points failed"
                    )
            
            # Special handling for equation solutions
            elif domain == "algebra" and problem_type in ["equation", "quadratic_equation", "polynomial_equation"]:
                return self._verify_equation_solution_numerical(problem, solution)
            
            else:
                return VerificationResult(
                    is_correct=False,
                    verification_method="numerical",
                    details={"error": "Missing required fields for numerical verification"},
                    confidence_score=0.0,
                    error_message="Cannot verify: missing expression or result"
                )
                
        except Exception as e:
            logger.error(f"Error in numerical verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="numerical",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Numerical verification failed: {e}"
            )
    
    def _verify_by_substitution(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any],
        domain: str
    ) -> VerificationResult:
        """
        Verify a solution by substituting it back into the original problem.
        
        Args:
            problem: Problem description
            solution: Solution to verify
            domain: Mathematical domain
            
        Returns:
            Verification result
        """
        problem_type = problem.get("problem_type", "")
        
        try:
            if domain == "algebra" and problem_type in ["equation", "quadratic_equation", "polynomial_equation"]:
                # Get the equation and solution
                if "equation" not in problem or "solutions" not in solution:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="substitution",
                        details={"error": "Missing equation or solutions"},
                        confidence_score=0.0,
                        error_message="Cannot verify: missing equation or solutions"
                    )
                
                # Parse the equation
                equation = self._parse_equation(problem["equation"])
                solutions = solution["solutions"]
                
                # If equation is in the form f(x) = g(x), convert to f(x) - g(x) = 0
                if isinstance(equation, sp.Eq):
                    equation_expr = equation.lhs - equation.rhs
                else:
                    equation_expr = equation
                
                # Get the variable from the equation
                if "variable" in problem:
                    var_name = problem["variable"]
                    var = sp.Symbol(var_name)
                elif hasattr(equation_expr, "free_symbols") and equation_expr.free_symbols:
                    var = list(equation_expr.free_symbols)[0]
                else:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="substitution",
                        details={"error": "Could not determine variable"},
                        confidence_score=0.0,
                        error_message="Cannot verify: could not determine variable"
                    )
                
                # Check each solution by substitution
                valid_solutions = 0
                invalid_solutions = []
                
                for sol in solutions:
                    # Parse the solution value
                    sol_value = self._parse_expression(sol)
                    
                    # Substitute into the equation
                    result = equation_expr.subs(var, sol_value)
                    
                    # Evaluate and check if approximately zero
                    try:
                        result_value = float(result.evalf())
                        is_valid = abs(result_value) < self.tolerance
                        
                        if is_valid:
                            valid_solutions += 1
                        else:
                            invalid_solutions.append({
                                "solution": str(sol),
                                "substitution_result": result_value
                            })
                    except Exception as e:
                        invalid_solutions.append({
                            "solution": str(sol),
                            "error": str(e)
                        })
                
                # Check if all solutions are valid
                is_correct = len(invalid_solutions) == 0
                confidence_score = valid_solutions / len(solutions) if solutions else 0.0
                
                return VerificationResult(
                    is_correct=is_correct,
                    verification_method="substitution",
                    details={
                        "total_solutions": len(solutions),
                        "valid_solutions": valid_solutions,
                        "invalid_solutions": invalid_solutions
                    },
                    confidence_score=confidence_score,
                    error_message=None if is_correct else f"{len(invalid_solutions)} invalid solutions found"
                )
            
            else:
                return VerificationResult(
                    is_correct=False,
                    verification_method="substitution",
                    details={"error": "Substitution verification not applicable for this problem type"},
                    confidence_score=0.0,
                    error_message="Substitution verification not applicable"
                )
                
        except Exception as e:
            logger.error(f"Error in substitution verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="substitution",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Substitution verification failed: {e}"
            )
    
    def _verify_equation_solution(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify the solution to an equation.
        
        Args:
            problem: Problem description including the equation
            solution: Solution containing the solutions
            
        Returns:
            Verification result
        """
        try:
            # Get the equation and solutions
            if "equation" not in problem or "solutions" not in solution:
                return VerificationResult(
                    is_correct=False,
                    verification_method="symbolic",
                    details={"error": "Missing equation or solutions"},
                    confidence_score=0.0,
                    error_message="Cannot verify: missing equation or solutions"
                )
            
            # Parse the equation
            equation = self._parse_equation(problem["equation"])
            
            # Get the expected solutions
            solution_values = []
            for sol in solution["solutions"]:
                try:
                    sol_expr = self._parse_expression(sol)
                    solution_values.append(sol_expr)
                except Exception as e:
                    logger.warning(f"Could not parse solution {sol}: {e}")
            
            # Solve the equation using SymPy
            if isinstance(equation, sp.Eq):
                var = list(equation.free_symbols)[0] if equation.free_symbols else None
            else:
                var = list(equation.free_symbols)[0] if hasattr(equation, "free_symbols") and equation.free_symbols else None
            
            if not var:
                return VerificationResult(
                    is_correct=False,
                    verification_method="symbolic",
                    details={"error": "Could not determine variable"},
                    confidence_score=0.0,
                    error_message="Cannot verify: could not determine variable"
                )
            
            # Solve symbolically
            try:
                if isinstance(equation, sp.Eq):
                    actual_solutions = sp.solve(equation, var)
                else:
                    actual_solutions = sp.solve(equation, var)
            except Exception as e:
                logger.error(f"Error solving equation: {e}")
                # If symbolic solving fails, rely on substitution
                return self._verify_by_substitution(problem, solution, "algebra")
            
            # Check if solutions match
            if not solution_values or not actual_solutions:
                # If either is empty, they should both be empty
                is_correct = len(solution_values) == len(actual_solutions)
                if is_correct:
                    return VerificationResult(
                        is_correct=True,
                        verification_method="symbolic",
                        details={
                            "expected_solutions": [str(s) for s in solution_values],
                            "actual_solutions": [str(s) for s in actual_solutions]
                        },
                        confidence_score=1.0,
                        error_message=None
                    )
                else:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="symbolic",
                        details={
                            "expected_solutions": [str(s) for s in solution_values],
                            "actual_solutions": [str(s) for s in actual_solutions]
                        },
                        confidence_score=0.0,
                        error_message="Solution count mismatch"
                    )
            
            # Check if each solution is in the actual solutions
            matches = 0
            mismatches = []
            
            for sol_value in solution_values:
                # Try to find a matching solution
                found_match = False
                for actual_sol in actual_solutions:
                    # Try symbolic comparison
                    try:
                        diff = sp.simplify(sol_value - actual_sol)
                        if diff == 0:
                            found_match = True
                            matches += 1
                            break
                    except Exception:
                        # If symbolic comparison fails, try numerical
                        try:
                            sol_num = float(sol_value.evalf())
                            actual_num = float(actual_sol.evalf())
                            if abs(sol_num - actual_num) < self.tolerance:
                                found_match = True
                                matches += 1
                                break
                        except Exception:
                            # If both comparisons fail, continue to next solution
                            pass
                
                if not found_match:
                    mismatches.append(str(sol_value))
            
            # Check if all solutions match
            is_correct = matches == len(solution_values)
            confidence_score = matches / len(solution_values) if solution_values else 0.0
            
            return VerificationResult(
                is_correct=is_correct,
                verification_method="symbolic",
                details={
                    "expected_solutions": [str(s) for s in solution_values],
                    "actual_solutions": [str(s) for s in actual_solutions],
                    "matching_solutions": matches,
                    "mismatches": mismatches
                },
                confidence_score=confidence_score,
                error_message=None if is_correct else f"{len(mismatches)} solutions do not match"
            )
                
        except Exception as e:
            logger.error(f"Error in equation solution verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="symbolic",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Equation solution verification failed: {e}"
            )
    
    def _verify_equation_solution_numerical(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify the solution to an equation using numerical methods.
        
        Args:
            problem: Problem description including the equation
            solution: Solution containing the solutions
            
        Returns:
            Verification result
        """
        try:
            # Get the equation and solutions
            if "equation" not in problem or "solutions" not in solution:
                return VerificationResult(
                    is_correct=False,
                    verification_method="numerical",
                    details={"error": "Missing equation or solutions"},
                    confidence_score=0.0,
                    error_message="Cannot verify: missing equation or solutions"
                )
            
            # Parse the equation
            equation = self._parse_equation(problem["equation"])
            
            # Get the variable from the equation
            if "variable" in problem:
                var_name = problem["variable"]
                var = sp.Symbol(var_name)
            elif isinstance(equation, sp.Eq) and equation.free_symbols:
                var = list(equation.free_symbols)[0]
            elif hasattr(equation, "free_symbols") and equation.free_symbols:
                var = list(equation.free_symbols)[0]
            else:
                return VerificationResult(
                    is_correct=False,
                    verification_method="numerical",
                    details={"error": "Could not determine variable"},
                    confidence_score=0.0,
                    error_message="Cannot verify: could not determine variable"
                )
            
            # Convert the equation to an expression equal to zero if needed
            if isinstance(equation, sp.Eq):
                equation_expr = equation.lhs - equation.rhs
            else:
                equation_expr = equation
            
            # Get the solutions
            solution_values = []
            for sol in solution["solutions"]:
                try:
                    sol_expr = self._parse_expression(sol)
                    # Evaluate to float if possible
                    try:
                        sol_value = float(sol_expr.evalf())
                    except Exception:
                        sol_value = sol_expr
                    solution_values.append(sol_value)
                except Exception as e:
                    logger.warning(f"Could not parse solution {sol}: {e}")
            
            # Check each solution by substitution
            valid_solutions = 0
            invalid_solutions = []
            
            for sol_value in solution_values:
                # Substitute into the equation
                try:
                    result = equation_expr.subs(var, sol_value)
                    # Evaluate and check if approximately zero
                    result_value = float(result.evalf())
                    is_valid = abs(result_value) < self.tolerance
                    
                    if is_valid:
                        valid_solutions += 1
                    else:
                        invalid_solutions.append({
                            "solution": str(sol_value),
                            "substitution_result": result_value
                        })
                except Exception as e:
                    invalid_solutions.append({
                        "solution": str(sol_value),
                        "error": str(e)
                    })
            
            # Check if all solutions are valid
            is_correct = len(invalid_solutions) == 0 and valid_solutions > 0
            confidence_score = valid_solutions / len(solution_values) if solution_values else 0.0
            
            return VerificationResult(
                is_correct=is_correct,
                verification_method="numerical",
                details={
                    "total_solutions": len(solution_values),
                    "valid_solutions": valid_solutions,
                    "invalid_solutions": invalid_solutions
                },
                confidence_score=confidence_score,
                error_message=None if is_correct else f"{len(invalid_solutions)} invalid solutions found"
            )
                
        except Exception as e:
            logger.error(f"Error in numerical equation verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="numerical",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Numerical equation verification failed: {e}"
            )
    
    def _verify_derivative(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify a derivative calculation.
        
        Args:
            problem: Problem description including the function and variable
            solution: Solution containing the derivative
            
        Returns:
            Verification result
        """
        try:
            # Get the function and variable
            if "function" not in problem or "derivative" not in solution:
                return VerificationResult(
                    is_correct=False,
                    verification_method="derivative",
                    details={"error": "Missing function or derivative"},
                    confidence_score=0.0,
                    error_message="Cannot verify: missing function or derivative"
                )
            
            # Parse the function and derivative
            function = self._parse_expression(problem["function"])
            derivative = self._parse_expression(solution["derivative"])
            
            # Get the variable of differentiation
            if "variable" in problem:
                var_name = problem["variable"]
                var = sp.Symbol(var_name)
            elif hasattr(function, "free_symbols") and function.free_symbols:
                var = list(function.free_symbols)[0]
            else:
                return VerificationResult(
                    is_correct=False,
                    verification_method="derivative",
                    details={"error": "Could not determine variable"},
                    confidence_score=0.0,
                    error_message="Cannot verify: could not determine variable"
                )
            
            # Calculate the derivative using SymPy
            expected_derivative = sp.diff(function, var)
            
            # Check if derivatives match
            difference = sp.simplify(derivative - expected_derivative)
            is_correct = difference == 0
            
            if not is_correct:
                # Try numerical comparison at sample points
                num_points = 5
                test_points = np.linspace(-5, 5, num_points)
                matches = 0
                
                for point in test_points:
                    try:
                        expected_value = float(expected_derivative.subs(var, point).evalf())
                        actual_value = float(derivative.subs(var, point).evalf())
                        
                        if abs(expected_value - actual_value) < self.tolerance:
                            matches += 1
                    except Exception:
                        continue
                
                # If all tested points match, consider it correct
                if matches == num_points:
                    is_correct = True
                    confidence_score = 0.9  # Slightly lower confidence for numerical verification
                else:
                    confidence_score = matches / num_points
            else:
                confidence_score = 1.0
            
            return VerificationResult(
                is_correct=is_correct,
                verification_method="derivative",
                details={
                    "function": str(function),
                    "expected_derivative": str(expected_derivative),
                    "provided_derivative": str(derivative),
                    "difference": str(difference)
                },
                confidence_score=confidence_score,
                error_message=None if is_correct else "Derivative does not match expected result"
            )
                
        except Exception as e:
            logger.error(f"Error in derivative verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="derivative",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Derivative verification failed: {e}"
            )
    
    def _verify_integral(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify an integration calculation.
        
        Args:
            problem: Problem description including the function and variable
            solution: Solution containing the integral
            
        Returns:
            Verification result
        """
        try:
            # Get the function and variable
            if "function" not in problem or "integral" not in solution:
                return VerificationResult(
                    is_correct=False,
                    verification_method="integral",
                    details={"error": "Missing function or integral"},
                    confidence_score=0.0,
                    error_message="Cannot verify: missing function or integral"
                )
            
            # Parse the function and integral
            function = self._parse_expression(problem["function"])
            integral = self._parse_expression(solution["integral"])
            
            # Get the variable of integration
            if "variable" in problem:
                var_name = problem["variable"]
                var = sp.Symbol(var_name)
            elif hasattr(function, "free_symbols") and function.free_symbols:
                var = list(function.free_symbols)[0]
            else:
                return VerificationResult(
                    is_correct=False,
                    verification_method="integral",
                    details={"error": "Could not determine variable"},
                    confidence_score=0.0,
                    error_message="Cannot verify: could not determine variable"
                )
            
            # Check if definite or indefinite integration
            if "limits" in problem:
                # Definite integration
                lower_limit = self._parse_expression(problem["limits"][0])
                upper_limit = self._parse_expression(problem["limits"][1])
                
                # Calculate the definite integral using SymPy
                expected_value = sp.integrate(function, (var, lower_limit, upper_limit))
                
                # Compare with the provided result
                try:
                    expected_float = float(expected_value.evalf())
                    integral_float = float(integral.evalf())
                    is_correct = abs(expected_float - integral_float) < self.tolerance
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="integral",
                        details={
                            "function": str(function),
                            "expected_value": expected_float,
                            "provided_value": integral_float,
                            "difference": abs(expected_float - integral_float)
                        },
                        confidence_score=1.0 if is_correct else 0.0,
                        error_message=None if is_correct else "Definite integral does not match expected result"
                    )
                except Exception:
                    # If numerical comparison fails, use symbolic
                    difference = sp.simplify(integral - expected_value)
                    is_correct = difference == 0
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="integral",
                        details={
                            "function": str(function),
                            "expected_value": str(expected_value),
                            "provided_value": str(integral),
                            "difference": str(difference)
                        },
                        confidence_score=1.0 if is_correct else 0.0,
                        error_message=None if is_correct else "Definite integral does not match expected result"
                    )
            
            else:
                # Indefinite integration - verify by differentiation
                # Strip any constant terms
                try:
                    # Calculate derivative of the integral
                    derivative = sp.diff(integral, var)
                    
                    # Check if it matches the original function
                    difference = sp.simplify(derivative - function)
                    is_correct = difference == 0
                    
                    if not is_correct:
                        # Try numerical comparison at sample points
                        num_points = 5
                        test_points = np.linspace(-5, 5, num_points)
                        matches = 0
                        
                        for point in test_points:
                            try:
                                derivative_value = float(derivative.subs(var, point).evalf())
                                function_value = float(function.subs(var, point).evalf())
                                
                                if abs(derivative_value - function_value) < self.tolerance:
                                    matches += 1
                            except Exception:
                                continue
                        
                        # If all tested points match, consider it correct
                        if matches == num_points:
                            is_correct = True
                            confidence_score = 0.9  # Slightly lower confidence for numerical verification
                        else:
                            confidence_score = matches / num_points
                    else:
                        confidence_score = 1.0
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="integral",
                        details={
                            "function": str(function),
                            "integral": str(integral),
                            "derivative_of_integral": str(derivative),
                            "difference": str(difference)
                        },
                        confidence_score=confidence_score,
                        error_message=None if is_correct else "Indefinite integral is incorrect"
                    )
                except Exception as e:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="integral",
                        details={"error": str(e)},
                        confidence_score=0.0,
                        error_message=f"Indefinite integral verification failed: {e}"
                    )
                
        except Exception as e:
            logger.error(f"Error in integral verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="integral",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Integral verification failed: {e}"
            )
    
    def _verify_limit(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify a limit calculation.
        
        Args:
            problem: Problem description including the function, variable, and limit point
            solution: Solution containing the limit value
            
        Returns:
            Verification result
        """
        try:
            # Get the function, variable, and limit point
            if "function" not in problem or "limit_value" not in solution or "limit_point" not in problem:
                return VerificationResult(
                    is_correct=False,
                    verification_method="limit",
                    details={"error": "Missing function, limit_value, or limit_point"},
                    confidence_score=0.0,
                    error_message="Cannot verify: missing function, limit_value, or limit_point"
                )
            
            # Parse the function and limit value
            function = self._parse_expression(problem["function"])
            limit_value = self._parse_expression(solution["limit_value"])
            limit_point = self._parse_expression(problem["limit_point"])
            
            # Get the variable
            if "variable" in problem:
                var_name = problem["variable"]
                var = sp.Symbol(var_name)
            elif hasattr(function, "free_symbols") and function.free_symbols:
                var = list(function.free_symbols)[0]
            else:
                return VerificationResult(
                    is_correct=False,
                    verification_method="limit",
                    details={"error": "Could not determine variable"},
                    confidence_score=0.0,
                    error_message="Cannot verify: could not determine variable"
                )
            
            # Get the direction if specified
            direction = problem.get("direction", None)
            
            # Calculate the limit using SymPy
            if direction == "left":
                expected_limit = sp.limit(function, var, limit_point, "-")
            elif direction == "right":
                expected_limit = sp.limit(function, var, limit_point, "+")
            else:
                expected_limit = sp.limit(function, var, limit_point)
            
            # Check if limits match
            try:
                # Try numerical comparison
                expected_float = float(expected_limit.evalf())
                limit_float = float(limit_value.evalf())
                is_correct = abs(expected_float - limit_float) < self.tolerance
                
                return VerificationResult(
                    is_correct=is_correct,
                    verification_method="limit",
                    details={
                        "function": str(function),
                        "expected_limit": expected_float,
                        "provided_limit": limit_float,
                        "difference": abs(expected_float - limit_float)
                    },
                    confidence_score=1.0 if is_correct else 0.0,
                    error_message=None if is_correct else "Limit does not match expected result"
                )
            except Exception:
                # If numerical comparison fails, use symbolic
                difference = sp.simplify(limit_value - expected_limit)
                is_correct = difference == 0
                
                return VerificationResult(
                    is_correct=is_correct,
                    verification_method="limit",
                    details={
                        "function": str(function),
                        "expected_limit": str(expected_limit),
                        "provided_limit": str(limit_value),
                        "difference": str(difference)
                    },
                    confidence_score=1.0 if is_correct else 0.0,
                    error_message=None if is_correct else "Limit does not match expected result"
                )
                
        except Exception as e:
            logger.error(f"Error in limit verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="limit",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Limit verification failed: {e}"
            )
    
    def _verify_matrix_operation(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify a matrix operation.
        
        Args:
            problem: Problem description including matrices and operation type
            solution: Solution containing the result matrix
            
        Returns:
            Verification result
        """
        try:
            # Get the operation type
            if "operation_type" not in problem or "result" not in solution:
                return VerificationResult(
                    is_correct=False,
                    verification_method="matrix",
                    details={"error": "Missing operation_type or result"},
                    confidence_score=0.0,
                    error_message="Cannot verify: missing operation_type or result"
                )
            
            operation_type = problem["operation_type"]
            
            # Parse the result
            result = solution["result"]
            
            # Handle different operation types
            if operation_type == "determinant":
                # Get the matrix
                if "matrix" not in problem:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="matrix",
                        details={"error": "Missing matrix for determinant calculation"},
                        confidence_score=0.0,
                        error_message="Cannot verify: missing matrix"
                    )
                
                # Parse the matrix
                matrix = self._parse_matrix(problem["matrix"])
                
                # Calculate the determinant
                expected_det = matrix.det()
                
                # Parse the result and compare
                try:
                    result_value = float(self._parse_expression(result).evalf())
                    expected_value = float(expected_det.evalf())
                    is_correct = abs(result_value - expected_value) < self.tolerance
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="matrix",
                        details={
                            "matrix": str(matrix),
                            "expected_determinant": expected_value,
                            "provided_determinant": result_value,
                            "difference": abs(result_value - expected_value)
                        },
                        confidence_score=1.0 if is_correct else 0.0,
                        error_message=None if is_correct else "Determinant does not match expected result"
                    )
                except Exception:
                    # If numerical comparison fails, use symbolic
                    result_expr = self._parse_expression(result)
                    difference = sp.simplify(result_expr - expected_det)
                    is_correct = difference == 0
                    
                    return VerificationResult(
                        is_correct=is_correct,
                        verification_method="matrix",
                        details={
                            "matrix": str(matrix),
                            "expected_determinant": str(expected_det),
                            "provided_determinant": str(result_expr),
                            "difference": str(difference)
                        },
                        confidence_score=1.0 if is_correct else 0.0,
                        error_message=None if is_correct else "Determinant does not match expected result"
                    )
            
            elif operation_type == "inverse":
                # Get the matrix
                if "matrix" not in problem:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="matrix",
                        details={"error": "Missing matrix for inverse calculation"},
                        confidence_score=0.0,
                        error_message="Cannot verify: missing matrix"
                    )
                
                # Parse the matrix and result
                matrix = self._parse_matrix(problem["matrix"])
                result_matrix = self._parse_matrix(result)
                
                # Verify the inverse by checking that A * A^(-1) = I
                product = matrix * result_matrix
                
                # Create identity matrix of the same size
                identity = sp.eye(matrix.rows)
                
                # Check if the product is the identity matrix
                is_correct = True
                for i in range(matrix.rows):
                    for j in range(matrix.rows):
                        expected = identity[i, j]
                        actual = product[i, j]
                        if abs(float(actual.evalf()) - float(expected.evalf())) >= self.tolerance:
                            is_correct = False
                            break
                
                return VerificationResult(
                    is_correct=is_correct,
                    verification_method="matrix",
                    details={
                        "matrix": str(matrix),
                        "provided_inverse": str(result_matrix),
                        "product": str(product),
                        "identity": str(identity)
                    },
                    confidence_score=1.0 if is_correct else 0.0,
                    error_message=None if is_correct else "Inverse does not satisfy A * A^(-1) = I"
                )
            
            elif operation_type in ["addition", "subtraction", "multiplication"]:
                # Get the matrices
                if "matrix_a" not in problem or "matrix_b" not in problem:
                    return VerificationResult(
                        is_correct=False,
                        verification_method="matrix",
                        details={"error": "Missing matrices for operation"},
                        confidence_score=0.0,
                        error_message="Cannot verify: missing matrices"
                    )
                
                # Parse the matrices and result
                matrix_a = self._parse_matrix(problem["matrix_a"])
                matrix_b = self._parse_matrix(problem["matrix_b"])
                result_matrix = self._parse_matrix(result)
                
                # Calculate expected result
                if operation_type == "addition":
                    expected_result = matrix_a + matrix_b
                elif operation_type == "subtraction":
                    expected_result = matrix_a - matrix_b
                else:  # multiplication
                    expected_result = matrix_a * matrix_b
                
                # Check if matrices are equal
                is_correct = True
                for i in range(expected_result.rows):
                    for j in range(expected_result.cols):
                        expected = expected_result[i, j]
                        actual = result_matrix[i, j]
                        if abs(float(actual.evalf()) - float(expected.evalf())) >= self.tolerance:
                            is_correct = False
                            break
                
                return VerificationResult(
                    is_correct=is_correct,
                    verification_method="matrix",
                    details={
                        "matrix_a": str(matrix_a),
                        "matrix_b": str(matrix_b),
                        "operation": operation_type,
                        "expected_result": str(expected_result),
                        "provided_result": str(result_matrix)
                    },
                    confidence_score=1.0 if is_correct else 0.0,
                    error_message=None if is_correct else f"Matrix {operation_type} result is incorrect"
                )
            
            else:
                return VerificationResult(
                    is_correct=False,
                    verification_method="matrix",
                    details={"error": f"Unsupported operation type: {operation_type}"},
                    confidence_score=0.0,
                    error_message=f"Cannot verify: unsupported operation type {operation_type}"
                )
                
        except Exception as e:
            logger.error(f"Error in matrix operation verification: {e}")
            return VerificationResult(
                is_correct=False,
                verification_method="matrix",
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Matrix operation verification failed: {e}"
            )
    
    def _parse_expression(self, expression) -> sp.Expr:
        """Parse a mathematical expression into a SymPy expression."""
        if isinstance(expression, sp.Expr):
            return expression
        
        try:
            return sp.sympify(expression)
        except Exception as e:
            logger.error(f"Error parsing expression '{expression}': {e}")
            raise ValueError(f"Could not parse expression: {expression}")
    
    def _parse_equation(self, equation) -> Union[sp.Eq, sp.Expr]:
        """Parse a mathematical equation into a SymPy equation."""
        if isinstance(equation, sp.Eq):
            return equation
        
        # Check if the string contains an equals sign
        if isinstance(equation, str) and "=" in equation:
            parts = equation.split("=", 1)
            left = self._parse_expression(parts[0].strip())
            right = self._parse_expression(parts[1].strip())
            return sp.Eq(left, right)
        
        # If no equals sign, treat as an expression
        return self._parse_expression(equation)
    
    def _parse_matrix(self, matrix) -> sp.Matrix:
        """Parse a matrix representation into a SymPy matrix."""
        if isinstance(matrix, sp.Matrix):
            return matrix
        
        try:
            # If matrix is a string representation of a nested list
            if isinstance(matrix, str):
                matrix = eval(matrix)
            
            # Convert to SymPy matrix
            return sp.Matrix(matrix)
        except Exception as e:
            logger.error(f"Error parsing matrix '{matrix}': {e}")
            raise ValueError(f"Could not parse matrix: {matrix}")

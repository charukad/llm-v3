"""
Linear Algebra Module - specialized linear algebra operations using SymPy.

This module implements linear algebra operations including matrix operations,
eigenvalue calculations, linear transformations, and vector spaces.
"""

import sympy as sp
from typing import Dict, List, Union, Any, Optional, Tuple
import logging


class LinearAlgebraProcessor:
    """Processor for linear algebra operations."""
    
    def __init__(self):
        """Initialize the linear algebra processor."""
        self.logger = logging.getLogger(__name__)
    
    def create_matrix(self, 
                     data: Union[List[List[float]], List[List[int]], str], 
                     symbolic: bool = False) -> Dict[str, Any]:
        """
        Create a matrix from data or string representation.
        
        Args:
            data: Matrix data as nested list or string representation
            symbolic: Whether to treat elements symbolically
            
        Returns:
            Dictionary with matrix information
        """
        try:
            matrix = None
            
            # Handle string input
            if isinstance(data, str):
                # Parse string to create matrix
                # Format should be like [[1, 2], [3, 4]]
                try:
                    # Strip whitespace and convert to SymPy Matrix
                    cleaned_data = data.strip()
                    matrix = sp.Matrix(sp.sympify(cleaned_data))
                except Exception as e:
                    return {
                        "success": False,
                        "matrix": None,
                        "error": f"Failed to parse matrix string: {str(e)}"
                    }
            else:
                # Create matrix from nested list
                matrix = sp.Matrix(data)
            
            # If matrix is successfully created, return information
            return {
                "success": True,
                "matrix": matrix,
                "shape": matrix.shape,
                "is_square": matrix.is_square,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error creating matrix: {str(e)}")
            return {
                "success": False,
                "matrix": None,
                "error": str(e)
            }
    
    def solve_linear_system(self, 
                          coefficients: Union[sp.Matrix, List[List[float]], str], 
                          constants: Union[List[float], sp.Matrix, str],
                          steps: bool = True) -> Dict[str, Any]:
        """
        Solve a system of linear equations.
        
        Args:
            coefficients: Coefficient matrix A in Ax = b
            constants: Constant vector b in Ax = b
            steps: Whether to generate steps
            
        Returns:
            Dictionary with solution information
        """
        try:
            # Create matrices if input is not already a SymPy Matrix
            A = coefficients
            if not isinstance(coefficients, sp.Matrix):
                A_result = self.create_matrix(coefficients)
                if not A_result["success"]:
                    return {
                        "success": False,
                        "solution": None,
                        "steps": None,
                        "error": A_result["error"]
                    }
                A = A_result["matrix"]
            
            b = constants
            if not isinstance(constants, sp.Matrix):
                if isinstance(constants, list):
                    b = sp.Matrix(constants)
                else:
                    # Try to parse as string
                    b = sp.Matrix(sp.sympify(constants))
            
            # Solve the system
            solution = A.solve(b)
            
            # Generate steps if requested
            if steps:
                solution_steps = self._generate_linear_system_steps(A, b, solution)
            else:
                solution_steps = []
            
            # Check for specific solution conditions
            conditions = self._analyze_linear_system(A, b, solution)
            
            return {
                "success": True,
                "solution": solution,
                "steps": solution_steps,
                "conditions": conditions,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error solving linear system: {str(e)}")
            return {
                "success": False,
                "solution": None,
                "steps": None,
                "error": str(e)
            }
    
    def calculate_determinant(self, 
                             matrix: Union[sp.Matrix, List[List[float]], str],
                             steps: bool = True) -> Dict[str, Any]:
        """
        Calculate the determinant of a matrix.
        
        Args:
            matrix: Input matrix
            steps: Whether to generate steps
            
        Returns:
            Dictionary with determinant information
        """
        try:
            # Create matrix if input is not already a SymPy Matrix
            A = matrix
            if not isinstance(matrix, sp.Matrix):
                A_result = self.create_matrix(matrix)
                if not A_result["success"]:
                    return {
                        "success": False,
                        "determinant": None,
                        "steps": None,
                        "error": A_result["error"]
                    }
                A = A_result["matrix"]
            
            # Check if matrix is square
            if not A.is_square:
                return {
                    "success": False,
                    "determinant": None,
                    "steps": None,
                    "error": "Matrix must be square to calculate determinant"
                }
            
            # Calculate determinant
            det = A.det()
            
            # Generate steps if requested
            if steps:
                det_steps = self._generate_determinant_steps(A, det)
            else:
                det_steps = []
            
            return {
                "success": True,
                "determinant": det,
                "steps": det_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error calculating determinant: {str(e)}")
            return {
                "success": False,
                "determinant": None,
                "steps": None,
                "error": str(e)
            }
    
    def calculate_eigenvalues(self, 
                            matrix: Union[sp.Matrix, List[List[float]], str],
                            steps: bool = True) -> Dict[str, Any]:
        """
        Calculate the eigenvalues and eigenvectors of a matrix.
        
        Args:
            matrix: Input matrix
            steps: Whether to generate steps
            
        Returns:
            Dictionary with eigenvalue information
        """
        try:
            # Create matrix if input is not already a SymPy Matrix
            A = matrix
            if not isinstance(matrix, sp.Matrix):
                A_result = self.create_matrix(matrix)
                if not A_result["success"]:
                    return {
                        "success": False,
                        "eigenvalues": None,
                        "eigenvectors": None,
                        "steps": None,
                        "error": A_result["error"]
                    }
                A = A_result["matrix"]
            
            # Check if matrix is square
            if not A.is_square:
                return {
                    "success": False,
                    "eigenvalues": None,
                    "eigenvectors": None,
                    "steps": None,
                    "error": "Matrix must be square to calculate eigenvalues"
                }
            
            # Calculate eigenvalues and eigenvectors
            eigensystem = A.eigenvects()
            
            # Format eigenvalues and eigenvectors
            eigenvalues = []
            eigenvectors = []
            
            for eigenvalue, multiplicity, basis in eigensystem:
                eigenvalues.append((eigenvalue, multiplicity))
                eigenvectors.append(basis)
            
            # Generate steps if requested
            if steps:
                eigen_steps = self._generate_eigenvalue_steps(A, eigensystem)
            else:
                eigen_steps = []
            
            return {
                "success": True,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "characteristic_polynomial": A.charpoly().as_expr(),
                "steps": eigen_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error calculating eigenvalues: {str(e)}")
            return {
                "success": False,
                "eigenvalues": None,
                "eigenvectors": None,
                "steps": None,
                "error": str(e)
            }
    
    def matrix_operations(self, 
                         operation: str,
                         matrix_a: Union[sp.Matrix, List[List[float]], str],
                         matrix_b: Optional[Union[sp.Matrix, List[List[float]], str]] = None,
                         scalar: Optional[Union[float, int, str]] = None,
                         steps: bool = True) -> Dict[str, Any]:
        """
        Perform operations on matrices.
        
        Args:
            operation: Type of operation (add, subtract, multiply, scalar_multiply, transpose, inverse)
            matrix_a: First input matrix
            matrix_b: Second input matrix (for binary operations)
            scalar: Scalar value (for scalar multiplication)
            steps: Whether to generate steps
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Create matrices if inputs are not already SymPy Matrices
            A = matrix_a
            if not isinstance(matrix_a, sp.Matrix):
                A_result = self.create_matrix(matrix_a)
                if not A_result["success"]:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": A_result["error"]
                    }
                A = A_result["matrix"]
            
            B = None
            if matrix_b is not None:
                if not isinstance(matrix_b, sp.Matrix):
                    B_result = self.create_matrix(matrix_b)
                    if not B_result["success"]:
                        return {
                            "success": False,
                            "result": None,
                            "steps": None,
                            "error": B_result["error"]
                        }
                    B = B_result["matrix"]
                else:
                    B = matrix_b
            
            # Convert scalar to SymPy object if provided as string
            if scalar is not None and isinstance(scalar, str):
                scalar = sp.sympify(scalar)
            
            # Perform the operation
            result = None
            operation_steps = []
            
            if operation == "add":
                if B is None:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": "Second matrix (B) is required for addition"
                    }
                
                # Check dimensions
                if A.shape != B.shape:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": f"Matrix dimensions don't match: A is {A.shape}, B is {B.shape}"
                    }
                
                result = A + B
                if steps:
                    operation_steps = self._generate_matrix_addition_steps(A, B, result)
                
            elif operation == "subtract":
                if B is None:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": "Second matrix (B) is required for subtraction"
                    }
                
                # Check dimensions
                if A.shape != B.shape:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": f"Matrix dimensions don't match: A is {A.shape}, B is {B.shape}"
                    }
                
                result = A - B
                if steps:
                    operation_steps = self._generate_matrix_subtraction_steps(A, B, result)
                
            elif operation == "multiply":
                if B is None:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": "Second matrix (B) is required for multiplication"
                    }
                
                # Check dimensions
                if A.shape[1] != B.shape[0]:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": f"Matrix dimensions incompatible for multiplication: A is {A.shape}, B is {B.shape}"
                    }
                
                result = A * B
                if steps:
                    operation_steps = self._generate_matrix_multiplication_steps(A, B, result)
                
            elif operation == "scalar_multiply":
                if scalar is None:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": "Scalar value is required for scalar multiplication"
                    }
                
                result = scalar * A
                if steps:
                    operation_steps = self._generate_scalar_multiplication_steps(scalar, A, result)
                
            elif operation == "transpose":
                result = A.transpose()
                if steps:
                    operation_steps = self._generate_transpose_steps(A, result)
                
            elif operation == "inverse":
                # Check if matrix is square
                if not A.is_square:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": "Matrix must be square to calculate inverse"
                    }
                
                # Check if matrix is invertible
                det = A.det()
                if det == 0:
                    return {
                        "success": False,
                        "result": None,
                        "steps": None,
                        "error": "Matrix is singular (determinant is zero), so it has no inverse"
                    }
                
                result = A.inv()
                if steps:
                    operation_steps = self._generate_inverse_steps(A, result)
                
            else:
                return {
                    "success": False,
                    "result": None,
                    "steps": None,
                    "error": f"Unsupported operation: {operation}"
                }
            
            return {
                "success": True,
                "result": result,
                "operation": operation,
                "steps": operation_steps,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error performing matrix operation: {str(e)}")
            return {
                "success": False,
                "result": None,
                "steps": None,
                "error": str(e)
            }
    
    def _generate_linear_system_steps(self, 
                                    A: sp.Matrix, 
                                    b: sp.Matrix, 
                                    solution: sp.Matrix) -> List[Dict[str, str]]:
        """
        Generate steps for solving a linear system.
        
        Args:
            A: Coefficient matrix
            b: Constant vector
            solution: System solution
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the system
        steps.append({
            "explanation": "Write the system of linear equations in matrix form Ax = b",
            "expression": "A = " + sp.latex(A) + ", \\quad b = " + sp.latex(b)
        })
        
        # Step 2: Augmented matrix
        augmented = A.row_join(b)
        steps.append({
            "explanation": "Create the augmented matrix [A|b]",
            "expression": sp.latex(augmented)
        })
        
        # Step 3: Row reduction (Gaussian elimination)
        reduced_row_echelon = A.row_join(b).rref()[0]
        steps.append({
            "explanation": "Perform Gaussian elimination to get the reduced row echelon form",
            "expression": sp.latex(reduced_row_echelon)
        })
        
        # Step 4: Extract solution
        steps.append({
            "explanation": "The solution to the system is",
            "expression": "x = " + sp.latex(solution)
        })
        
        return steps
    
    def _generate_determinant_steps(self, 
                                  A: sp.Matrix, 
                                  det: Union[sp.Expr, int, float]) -> List[Dict[str, str]]:
        """
        Generate steps for calculating a determinant.
        
        Args:
            A: Input matrix
            det: Calculated determinant
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrix
        steps.append({
            "explanation": "Calculate the determinant of the matrix",
            "expression": "A = " + sp.latex(A)
        })
        
        # For 2x2 and 3x3 matrices, show explicit formula
        n = A.shape[0]
        
        if n == 2:
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            steps.append({
                "explanation": "For a 2×2 matrix, we use the formula: ad - bc",
                "expression": f"|A| = ({sp.latex(a)})({sp.latex(d)}) - ({sp.latex(b)})({sp.latex(c)})"
            })
            steps.append({
                "explanation": "Calculate the determinant",
                "expression": f"|A| = {sp.latex(det)}"
            })
        elif n == 3:
            steps.append({
                "explanation": "For a 3×3 matrix, we use the formula with cofactor expansion or the rule of Sarrus",
                "expression": "\\text{(Cofactor expansion steps omitted for brevity)}"
            })
            steps.append({
                "explanation": "Calculate the determinant",
                "expression": f"|A| = {sp.latex(det)}"
            })
        else:
            steps.append({
                "explanation": "For matrices larger than 3×3, we typically use cofactor expansion or row reduction",
                "expression": "\\text{(Calculation steps omitted for brevity)}"
            })
            steps.append({
                "explanation": "The determinant evaluates to",
                "expression": f"|A| = {sp.latex(det)}"
            })
        
        return steps
    
    def _generate_eigenvalue_steps(self, 
                                 A: sp.Matrix, 
                                 eigensystem: List[Tuple]) -> List[Dict[str, str]]:
        """
        Generate steps for calculating eigenvalues and eigenvectors.
        
        Args:
            A: Input matrix
            eigensystem: Calculated eigenvalues and eigenvectors
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrix
        steps.append({
            "explanation": "Find the eigenvalues and eigenvectors of the matrix",
            "expression": "A = " + sp.latex(A)
        })
        
        # Step 2: Characteristic polynomial
        charpoly = A.charpoly().as_expr()
        steps.append({
            "explanation": "Calculate the characteristic polynomial: det(A - λI)",
            "expression": "p(\\lambda) = " + sp.latex(charpoly)
        })
        
        # Step 3: Eigenvalues (roots of the characteristic polynomial)
        eigenvalues_text = ""
        for eigenvalue, multiplicity, _ in eigensystem:
            if multiplicity > 1:
                eigenvalues_text += f"\\lambda = {sp.latex(eigenvalue)} \\text{{ (multiplicity {multiplicity})}}; "
            else:
                eigenvalues_text += f"\\lambda = {sp.latex(eigenvalue)}; "
        
        eigenvalues_text = eigenvalues_text.rstrip("; ")
        steps.append({
            "explanation": "Find the eigenvalues by solving p(λ) = 0",
            "expression": eigenvalues_text
        })
        
        # Step 4: Eigenvectors
        for i, (eigenvalue, multiplicity, basis) in enumerate(eigensystem):
            steps.append({
                "explanation": f"Find the eigenvectors for λ = {sp.latex(eigenvalue)}",
                "expression": f"\\text{{Solve }} (A - {sp.latex(eigenvalue)}I)v = 0"
            })
            
            eigenvector_text = ""
            for j, vector in enumerate(basis):
                eigenvector_text += f"v_{j+1} = {sp.latex(vector)}; "
            
            eigenvector_text = eigenvector_text.rstrip("; ")
            steps.append({
                "explanation": f"Eigenvectors for λ = {sp.latex(eigenvalue)}",
                "expression": eigenvector_text
            })
        
        return steps
    
    def _generate_matrix_addition_steps(self, 
                                      A: sp.Matrix, 
                                      B: sp.Matrix, 
                                      result: sp.Matrix) -> List[Dict[str, str]]:
        """
        Generate steps for matrix addition.
        
        Args:
            A: First matrix
            B: Second matrix
            result: Result matrix
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrices
        steps.append({
            "explanation": "Add the matrices A and B",
            "expression": "A = " + sp.latex(A) + ", \\quad B = " + sp.latex(B)
        })
        
        # Step 2: Explanation of the process
        steps.append({
            "explanation": "Matrix addition is performed element-wise",
            "expression": "(A + B)_{ij} = A_{ij} + B_{ij}"
        })
        
        # Step 3: Show the result
        steps.append({
            "explanation": "The result of A + B is",
            "expression": sp.latex(result)
        })
        
        return steps
    
    def _generate_matrix_subtraction_steps(self, 
                                        A: sp.Matrix, 
                                        B: sp.Matrix, 
                                        result: sp.Matrix) -> List[Dict[str, str]]:
        """
        Generate steps for matrix subtraction.
        
        Args:
            A: First matrix
            B: Second matrix
            result: Result matrix
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrices
        steps.append({
            "explanation": "Subtract matrix B from matrix A",
            "expression": "A = " + sp.latex(A) + ", \\quad B = " + sp.latex(B)
        })
        
        # Step 2: Explanation of the process
        steps.append({
            "explanation": "Matrix subtraction is performed element-wise",
            "expression": "(A - B)_{ij} = A_{ij} - B_{ij}"
        })
        
        # Step 3: Show the result
        steps.append({
            "explanation": "The result of A - B is",
            "expression": sp.latex(result)
        })
        
        return steps
    
    def _generate_matrix_multiplication_steps(self, 
                                           A: sp.Matrix, 
                                           B: sp.Matrix, 
                                           result: sp.Matrix) -> List[Dict[str, str]]:
        """
        Generate steps for matrix multiplication.
        
        Args:
            A: First matrix
            B: Second matrix
            result: Result matrix
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrices
        steps.append({
            "explanation": "Multiply matrices A and B",
            "expression": "A = " + sp.latex(A) + ", \\quad B = " + sp.latex(B)
        })
        
        # Step 2: Explanation of the process
        steps.append({
            "explanation": "Matrix multiplication: (AB)_{ij} = sum_k(A_{ik} * B_{kj})",
            "expression": "\\text{Dimensions: } A \\text{ is } " + str(A.shape[0]) + " \\times " + str(A.shape[1]) + ", B \\text{ is } " + str(B.shape[0]) + " \\times " + str(B.shape[1])
        })
        
        # Step 3: Show the result
        steps.append({
            "explanation": "The result of AB is",
            "expression": sp.latex(result)
        })
        
        return steps
    
    def _generate_scalar_multiplication_steps(self, 
                                          scalar: Union[float, int, sp.Expr], 
                                          A: sp.Matrix, 
                                          result: sp.Matrix) -> List[Dict[str, str]]:
        """
        Generate steps for scalar multiplication.
        
        Args:
            scalar: Scalar value
            A: Input matrix
            result: Result matrix
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrix and scalar
        steps.append({
            "explanation": "Multiply matrix A by scalar c",
            "expression": "c = " + sp.latex(scalar) + ", \\quad A = " + sp.latex(A)
        })
        
        # Step 2: Explanation of the process
        steps.append({
            "explanation": "Scalar multiplication is performed element-wise",
            "expression": "(cA)_{ij} = c \\cdot A_{ij}"
        })
        
        # Step 3: Show the result
        steps.append({
            "explanation": "The result of cA is",
            "expression": sp.latex(result)
        })
        
        return steps
    
    def _generate_transpose_steps(self, 
                               A: sp.Matrix, 
                               result: sp.Matrix) -> List[Dict[str, str]]:
        """
        Generate steps for matrix transposition.
        
        Args:
            A: Input matrix
            result: Transposed matrix
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrix
        steps.append({
            "explanation": "Find the transpose of matrix A",
            "expression": "A = " + sp.latex(A)
        })
        
        # Step 2: Explanation of the process
        steps.append({
            "explanation": "The transpose of a matrix involves swapping rows and columns",
            "expression": "(A^T)_{ij} = A_{ji}"
        })
        
        # Step 3: Show the result
        steps.append({
            "explanation": "The transpose of A is",
            "expression": "A^T = " + sp.latex(result)
        })
        
        return steps
    
    def _generate_inverse_steps(self, 
                             A: sp.Matrix, 
                             result: sp.Matrix) -> List[Dict[str, str]]:
        """
        Generate steps for matrix inversion.
        
        Args:
            A: Input matrix
            result: Inverse matrix
            
        Returns:
            List of steps as dictionaries with explanation and expression
        """
        steps = []
        
        # Step 1: Display the matrix
        steps.append({
            "explanation": "Find the inverse of matrix A",
            "expression": "A = " + sp.latex(A)
        })
        
        # Step 2: Check invertibility
        det = A.det()
        steps.append({
            "explanation": "Check if the matrix is invertible by calculating its determinant",
            "expression": "|A| = " + sp.latex(det)
        })
        
        if det != 0:
            steps.append({
                "explanation": "Since the determinant is non-zero, the matrix is invertible",
                "expression": ""
            })
        else:
            steps.append({
                "explanation": "Since the determinant is zero, the matrix is not invertible",
                "expression": ""
            })
            return steps
        
        # Step 3: Explain inversion process
        if A.shape[0] == 2:
            # For 2x2 matrices, show the explicit formula
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            steps.append({
                "explanation": "For a 2×2 matrix, we use the formula: 1/det(A) * [[d, -b], [-c, a]]",
                "expression": "A^{-1} = \\frac{1}{" + sp.latex(det) + "} \\begin{bmatrix} " + sp.latex(d) + " & " + sp.latex(-b) + " \\\\ " + sp.latex(-c) + " & " + sp.latex(a) + " \\end{bmatrix}"
            })
        else:
            # For larger matrices, just mention the methods
            steps.append({
                "explanation": "For larger matrices, inversion can be performed using the adjugate method or Gaussian elimination",
                "expression": "\\text{(Calculation steps omitted for brevity)}"
            })
        
        # Step 4: Show the result
        steps.append({
            "explanation": "The inverse of A is",
            "expression": "A^{-1} = " + sp.latex(result)
        })
        
        return steps
    
    def _analyze_linear_system(self, 
                             A: sp.Matrix, 
                             b: sp.Matrix, 
                             solution: sp.Matrix) -> Dict[str, Any]:
        """
        Analyze a linear system for specific conditions.
        
        Args:
            A: Coefficient matrix
            b: Constant vector
            solution: System solution
            
        Returns:
            Dictionary with condition information
        """
        conditions = {}
        
        # Check if the system is consistent
        augmented = A.row_join(b)
        rank_A = A.rank()
        rank_augmented = augmented.rank()
        
        if rank_A == rank_augmented:
            conditions["consistent"] = True
        else:
            conditions["consistent"] = False
            conditions["message"] = "The system is inconsistent (no solution exists)"
            return conditions
        
        # Check for unique or infinite solutions
        if rank_A == A.shape[1]:
            conditions["unique_solution"] = True
            conditions["message"] = "The system has a unique solution"
        else:
            conditions["unique_solution"] = False
            conditions["message"] = "The system has infinitely many solutions"
            conditions["free_variables"] = A.shape[1] - rank_A
        
        return conditions

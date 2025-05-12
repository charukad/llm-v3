"""
Evaluation framework for mathematical capabilities.

This module provides tools to evaluate the mathematical reasoning and computation
capabilities of the Core LLM Agent across different domains.
"""
import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
import re

logger = logging.getLogger(__name__)

class MathEvaluator:
    """Evaluator for mathematical capabilities of the LLM system."""
    
    def __init__(self, llm_agent, sympy_processor=None):
        """
        Initialize the evaluator.
        
        Args:
            llm_agent: Core LLM Agent to evaluate
            sympy_processor: Optional SymPy processor for verification
        """
        self.llm_agent = llm_agent
        self.sympy_processor = sympy_processor
        
        # Load evaluation datasets
        self.datasets = self._load_evaluation_datasets()
    
    def _load_evaluation_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load evaluation datasets for different mathematical domains.
        
        Returns:
            Dictionary mapping domains to lists of evaluation examples
        """
        # This is a placeholder for actual dataset loading
        # In a real implementation, you would load these from files
        
        datasets = {
            "algebra": [
                {
                    "id": "algebra-001",
                    "question": "Solve for x: 2x + 3 = 7",
                    "reference_answer": "x = 2",
                    "solution_type": "exact",
                    "difficulty": "easy"
                },
                {
                    "id": "algebra-002",
                    "question": "Solve the quadratic equation: x² - 5x + 6 = 0",
                    "reference_answer": "x = 2 or x = 3",
                    "solution_type": "exact",
                    "difficulty": "medium"
                }
            ],
            "calculus": [
                {
                    "id": "calculus-001",
                    "question": "Find the derivative of f(x) = x² + 2x + 1",
                    "reference_answer": "f'(x) = 2x + 2",
                    "solution_type": "exact",
                    "difficulty": "easy"
                },
                {
                    "id": "calculus-002",
                    "question": "Calculate the definite integral: ∫₀¹ x² dx",
                    "reference_answer": "1/3",
                    "solution_type": "exact",
                    "difficulty": "medium"
                }
            ]
        }
        
        return datasets
    
    def extract_final_answer(self, response: str) -> str:
        """
        Extract the final answer from a verbose response.
        
        Args:
            response: Full response text
            
        Returns:
            Extracted final answer
        """
        # Look for specific patterns that indicate a final answer
        patterns = [
            r"Therefore,\s*(.*?)\s*$",  # Therefore, x = 2
            r"Thus,\s*(.*?)\s*$",       # Thus, x = 2
            r"The\s+(?:final\s+)?answer\s+is\s+(.*?)\s*[\.;]", # The answer is x = 2.
            r"(?:We|I)\s+(?:get|find|have)\s+(.*?)\s*$",  # We get x = 2
            r"(?:So|Hence),\s*(.*?)\s*$"  # So, x = 2
        ]
        
        response_lines = response.strip().split("\n")
        # Check the last few lines first (where the answer is most likely to be)
        for line in reversed(response_lines[-5:]):  
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1).strip()
        
        # If no patterns match, return the last line as a fallback
        return response_lines[-1].strip()
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer for comparison.
        
        Args:
            answer: Answer string to normalize
            
        Returns:
            Normalized answer string
        """
        # Remove spaces around operators and equals signs
        answer = re.sub(r'\s*([=+\-*/^()])\s*', r'\1', answer)
        
        # Replace multiple spaces with a single space
        answer = re.sub(r'\s+', ' ', answer)
        
        # Standardize alternative forms of "or"
        answer = re.sub(r'\s*(?:or|OR|Or)\s*', ' or ', answer)
        
        # Standardize "x = 2 or x = 3" to just "x = 2 or 3"
        answer = re.sub(r'([a-zA-Z])\s*=\s*([^=\s]+)\s+or\s+\1\s*=\s*', r'\1 = \2 or ', answer)
        
        # Remove "the answer is" and similar phrases
        answer = re.sub(r'^(?:the answer is|we get|we have|therefore|thus|hence|so)[,:]?\s*', '', answer, flags=re.I)
        
        # Remove trailing punctuation
        answer = re.sub(r'[.;,]$', '', answer)
        
        return answer.strip()
    
    def compare_answers(self, generated: str, reference: str) -> float:
        """
        Compare a generated answer to a reference answer.
        
        Args:
            generated: Answer generated by the model
            reference: Reference correct answer
            
        Returns:
            Score between 0 and 1 indicating correctness
        """
        # Normalize both answers
        norm_generated = self.normalize_answer(generated)
        norm_reference = self.normalize_answer(reference)
        
        # Exact match check
        if norm_generated == norm_reference:
            return 1.0
        
        # Alternative forms of equivalent answers
        # For example, "x = 2 or x = 3" vs "x = 3 or x = 2"
        if "or" in norm_reference and "or" in norm_generated:
            gen_parts = set(part.strip() for part in norm_generated.split("or"))
            ref_parts = set(part.strip() for part in norm_reference.split("or"))
            if gen_parts == ref_parts:
                return 1.0
        
        # If we had a symbolic math processor, we could check mathematical equivalence
        # This would be implemented if sympy_processor was provided
        
        # Partial credit for close answers
        # This is a simplified approach, in a real system you would use more
        # sophisticated fuzzy matching and mathematical equivalence checking
        if norm_reference in norm_generated or norm_generated in norm_reference:
            return 0.8
        
        # Split the answers and check for overlapping parts
        ref_terms = set(norm_reference.split())
        gen_terms = set(norm_generated.split())
        overlap = len(ref_terms.intersection(gen_terms))
        
        if overlap > 0:
            similarity = overlap / max(len(ref_terms), len(gen_terms))
            return min(0.6, similarity)  # Cap partial credit at 0.6
        
        return 0.0  # No match
    
    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single example.
        
        Args:
            example: Example dictionary with question and reference_answer
            
        Returns:
            Evaluation results including scores and timings
        """
        question = example["question"]
        reference_answer = example["reference_answer"]
        
        # Track evaluation metrics
        start_time = time.time()
        
        # Generate response
        response = self.llm_agent.generate_response(
            prompt=question,
            domain=example.get("domain"),
            use_cot=True
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Extract the final answer from the response
        extracted_answer = self.extract_final_answer(response)
        
        # Compare to reference answer
        score = self.compare_answers(extracted_answer, reference_answer)
        
        # Create evaluation result
        result = {
            "id": example.get("id", "unknown"),
            "question": question,
            "reference_answer": reference_answer,
            "model_response": response,
            "extracted_answer": extracted_answer,
            "normalized_answer": self.normalize_answer(extracted_answer),
            "score": score,
            "generation_time": generation_time,
            "domain": example.get("domain"),
            "difficulty": example.get("difficulty", "medium")
        }
        
        return result
    
    def evaluate_domain(self, domain: str, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate all examples for a specific domain.
        
        Args:
            domain: Mathematical domain to evaluate
            max_examples: Maximum number of examples to evaluate
            
        Returns:
            Evaluation results for the domain
        """
        if domain not in self.datasets:
            logger.warning(f"No evaluation dataset found for domain: {domain}")
            return {"error": f"No evaluation dataset for domain: {domain}"}
        
        examples = self.datasets[domain]
        if max_examples is not None:
            examples = examples[:max_examples]
        
        logger.info(f"Evaluating {len(examples)} examples for domain: {domain}")
        
        results = []
        total_score = 0.0
        total_time = 0.0
        
        for example in examples:
            example["domain"] = domain  # Ensure domain is set
            result = self.evaluate_example(example)
            results.append(result)
            
            total_score += result["score"]
            total_time += result["generation_time"]
        
        average_score = total_score / len(examples) if examples else 0
        average_time = total_time / len(examples) if examples else 0
        
        domain_results = {
            "domain": domain,
            "num_examples": len(examples),
            "average_score": average_score,
            "average_generation_time": average_time,
            "results": results
        }
        
        logger.info(f"Domain {domain} evaluation complete. " 
                   f"Average score: {average_score:.2f}, " 
                   f"Average time: {average_time:.2f}s")
        
        return domain_results
    
    def evaluate_all_domains(self, max_examples_per_domain: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate all available domains.
        
        Args:
            max_examples_per_domain: Maximum examples per domain
            
        Returns:
            Comprehensive evaluation results
        """
        all_results = {}
        overall_score = 0.0
        total_examples = 0
        
        for domain in self.datasets.keys():
            domain_result = self.evaluate_domain(domain, max_examples_per_domain)
            all_results[domain] = domain_result
            
            # Update overall statistics
            if "error" not in domain_result:
                overall_score += domain_result["average_score"] * domain_result["num_examples"]
                total_examples += domain_result["num_examples"]
        
        # Calculate overall average
        final_score = overall_score / total_examples if total_examples > 0 else 0
        
        evaluation_summary = {
            "overall_score": final_score,
            "total_examples": total_examples,
            "domains_evaluated": len(all_results),
            "domain_results": all_results
        }
        
        logger.info(f"Evaluation complete. Overall score: {final_score:.2f} across {total_examples} examples")
        
        return evaluation_summary
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results to save
            output_path: Path to save results to
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

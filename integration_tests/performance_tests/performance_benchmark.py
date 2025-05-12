"""
Comprehensive performance benchmarking system for the Mathematical Multimodal LLM.
Measures performance across different components and under various load conditions.
"""
import time
import statistics
import asyncio
import argparse
import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional
from datetime import datetime
import concurrent.futures
import requests
from tqdm import tqdm

# Performance benchmark configuration
DEFAULT_CONFIG = {
    "api_base_url": "http://localhost:8000/api",
    "iterations": 10,
    "concurrent_users": [1, 5, 10, 20, 50],
    "test_cases": [
        {
            "name": "simple_arithmetic",
            "query": "What is 125 * 37?",
            "expected_results": ["4625"]
        },
        {
            "name": "algebra_equation",
            "query": "Solve the equation 3x + 7 = 22",
            "expected_results": ["x = 5"]
        },
        {
            "name": "calculus_derivative",
            "query": "Find the derivative of x^3 * sin(x)",
            "expected_results": ["3x^2", "sin(x)", "cos(x)"]
        },
        {
            "name": "calculus_integral",
            "query": "Calculate the integral of x^2 from 0 to 3",
            "expected_results": ["9"]
        },
        {
            "name": "linear_algebra",
            "query": "Find the determinant of [[4, 2], [3, 1]]",
            "expected_results": ["-2"]
        },
        {
            "name": "visualization",
            "query": "Plot the function f(x) = x^2 - 4x + 4 for x from -2 to 6",
            "expected_results": [],
            "check_visualization": True
        }
    ],
    "output_dir": "benchmark_results",
    "generate_plots": True
}

class PerformanceBenchmark:
    """
    Performance benchmarking system for measuring system performance
    across different components and workloads.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the benchmarking system with configuration.
        
        Args:
            config: Dictionary containing benchmark configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.results = {}
        self.api_base_url = self.config["api_base_url"]
        self.output_dir = self.config["output_dir"]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def run_benchmarks(self):
        """Run all benchmarks according to configuration."""
        print(f"Starting performance benchmarks at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API Base URL: {self.api_base_url}")
        
        # First, run component-specific benchmarks
        await self.benchmark_individual_components()
        
        # Then, run end-to-end benchmarks with different concurrency levels
        for concurrent_users in self.config["concurrent_users"]:
            print(f"\nRunning end-to-end benchmarks with {concurrent_users} concurrent users...")
            await self.benchmark_end_to_end(concurrent_users)
        
        # Generate reports
        self.generate_reports()
        
        print(f"\nBenchmarks completed. Results saved to {self.output_dir}")
    
    async def benchmark_individual_components(self):
        """Benchmark individual system components."""
        print("\nBenchmarking individual components...")
        
        components = [
            ("llm_inference", self.benchmark_llm_inference),
            ("mathematical_computation", self.benchmark_mathematical_computation),
            ("visualization_generation", self.benchmark_visualization_generation),
            ("database_operations", self.benchmark_database_operations),
            ("response_generation", self.benchmark_response_generation)
        ]
        
        for component_name, benchmark_func in components:
            print(f"  Benchmarking {component_name}...")
            component_results = await benchmark_func()
            self.results[component_name] = component_results
    
    async def benchmark_llm_inference(self) -> Dict[str, Any]:
        """Benchmark LLM inference performance."""
        # Use simple prompts to test raw inference speed
        prompts = [
            "What is the capital of France?",
            "Explain the Pythagorean theorem.",
            "List three properties of prime numbers.",
            "What is the formula for the area of a circle?",
            "Describe Newton's laws of motion."
        ]
        
        url = f"{self.api_base_url}/benchmark/llm_inference"
        
        durations = []
        for prompt in prompts:
            for _ in range(self.config["iterations"]):
                start_time = time.time()
                response = requests.post(url, json={"prompt": prompt})
                
                if response.status_code != 200:
                    print(f"Error benchmarking LLM inference: {response.text}")
                    continue
                
                duration = time.time() - start_time
                durations.append(duration)
        
        return self._calculate_statistics(durations)
    
    async def benchmark_mathematical_computation(self) -> Dict[str, Any]:
        """Benchmark mathematical computation performance."""
        computations = [
            {"expression": "x^2 + 3*x + 2", "operation": "differentiate", "variable": "x"},
            {"expression": "x^3", "operation": "integrate", "variable": "x"},
            {"expression": "3*x + 7 = 22", "operation": "solve", "variable": "x"},
            {"expression": "x^2 - 4*x + 4", "operation": "factor", "variable": "x"},
            {"expression": "[[1, 2], [3, 4]]", "operation": "determinant"}
        ]
        
        url = f"{self.api_base_url}/benchmark/math_computation"
        
        durations = []
        for computation in computations:
            for _ in range(self.config["iterations"]):
                start_time = time.time()
                response = requests.post(url, json=computation)
                
                if response.status_code != 200:
                    print(f"Error benchmarking mathematical computation: {response.text}")
                    continue
                
                duration = time.time() - start_time
                durations.append(duration)
        
        return self._calculate_statistics(durations)
    
    async def benchmark_visualization_generation(self) -> Dict[str, Any]:
        """Benchmark visualization generation performance."""
        visualizations = [
            {"function": "sin(x)", "x_range": [-3.14, 3.14], "type": "2d"},
            {"function": "x^2", "x_range": [-5, 5], "type": "2d"},
            {"function": "sin(x)*cos(x)", "x_range": [-3.14, 3.14], "type": "2d"},
            {"function": "sin(sqrt(x^2 + y^2))", "x_range": [-5, 5], "y_range": [-5, 5], "type": "3d"},
            {"function": "x^2 + y^2", "x_range": [-2, 2], "y_range": [-2, 2], "type": "3d"}
        ]
        
        url = f"{self.api_base_url}/benchmark/visualization"
        
        durations = []
        for viz in visualizations:
            for _ in range(self.config["iterations"]):
                start_time = time.time()
                response = requests.post(url, json=viz)
                
                if response.status_code != 200:
                    print(f"Error benchmarking visualization generation: {response.text}")
                    continue
                
                duration = time.time() - start_time
                durations.append(duration)
        
        return self._calculate_statistics(durations)
    
    async def benchmark_database_operations(self) -> Dict[str, Any]:
        """Benchmark database operations performance."""
        operations = [
            {"type": "read", "collection": "conversations", "query": {"limit": 10}},
            {"type": "read", "collection": "expressions", "query": {"limit": 10}},
            {"type": "write", "collection": "test_benchmark", "data": {"test": "data", "timestamp": "now"}},
            {"type": "aggregate", "collection": "interactions", "pipeline": [{"$group": {"_id": "$user_id", "count": {"$sum": 1}}}]},
            {"type": "index_scan", "collection": "math_expressions", "query": {"domain": "calculus"}}
        ]
        
        url = f"{self.api_base_url}/benchmark/database"
        
        durations = []
        for operation in operations:
            for _ in range(self.config["iterations"]):
                start_time = time.time()
                response = requests.post(url, json=operation)
                
                if response.status_code != 200:
                    print(f"Error benchmarking database operations: {response.text}")
                    continue
                
                duration = time.time() - start_time
                durations.append(duration)
        
        # Cleanup test data
        requests.post(f"{self.api_base_url}/benchmark/database_cleanup")
        
        return self._calculate_statistics(durations)
    
    async def benchmark_response_generation(self) -> Dict[str, Any]:
        """Benchmark end-to-end response generation (without LLM)."""
        queries = [
            "What is the derivative of x^2?",
            "Integrate sin(x) with respect to x",
            "Solve the equation x^2 = 9",
            "Calculate the determinant of [[1, 2], [3, 4]]",
            "Find the eigenvalues of [[4, 2], [1, 3]]"
        ]
        
        url = f"{self.api_base_url}/benchmark/response_generation"
        
        durations = []
        for query in queries:
            for _ in range(self.config["iterations"]):
                start_time = time.time()
                response = requests.post(url, json={"query": query, "skip_llm": True})
                
                if response.status_code != 200:
                    print(f"Error benchmarking response generation: {response.text}")
                    continue
                
                duration = time.time() - start_time
                durations.append(duration)
        
        return self._calculate_statistics(durations)
    
    async def benchmark_end_to_end(self, concurrent_users: int):
        """
        Benchmark end-to-end system performance with multiple concurrent users.
        
        Args:
            concurrent_users: Number of concurrent users to simulate
        """
        test_cases = self.config["test_cases"]
        iterations = self.config["iterations"]
        
        # Results structure for this concurrency level
        results_key = f"end_to_end_{concurrent_users}_users"
        self.results[results_key] = {
            "concurrent_users": concurrent_users,
            "total_requests": len(test_cases) * iterations,
            "test_cases": {}
        }
        
        for test_case in test_cases:
            print(f"  Running test case: {test_case['name']} with {concurrent_users} concurrent users")
            
            # Structure to store results for this test case
            case_results = {
                "durations": [],
                "success_rate": 0,
                "errors": 0,
                "throughput": 0
            }
            
            async def run_single_request():
                url = f"{self.api_base_url}/math/query"
                payload = {
                    "query": test_case["query"],
                    "require_steps": True,
                    "require_visualization": test_case.get("check_visualization", False)
                }
                
                try:
                    start_time = time.time()
                    response = requests.post(url, json=payload)
                    duration = time.time() - start_time
                    
                    if response.status_code != 200:
                        case_results["errors"] += 1
                        return False, duration
                    
                    # Verify expected results if provided
                    result_text = response.json().get("response", "")
                    expected_results = test_case.get("expected_results", [])
                    
                    if expected_results:
                        found_all = all(expected in result_text for expected in expected_results)
                        if not found_all:
                            case_results["errors"] += 1
                            return False, duration
                    
                    # Check for visualization if required
                    if test_case.get("check_visualization", False):
                        has_viz = (
                            "visualization_urls" in response.json() and 
                            len(response.json()["visualization_urls"]) > 0
                        )
                        if not has_viz:
                            case_results["errors"] += 1
                            return False, duration
                    
                    case_results["durations"].append(duration)
                    return True, duration
                    
                except Exception as e:
                    print(f"Error during request: {str(e)}")
                    case_results["errors"] += 1
                    return False, 0
            
            # Run concurrent requests
            total_tasks = iterations
            tasks = []
            
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async def bounded_run_request():
                async with semaphore:
                    return await run_single_request()
            
            start_time = time.time()
            
            # Create all tasks
            for _ in range(total_tasks):
                tasks.append(asyncio.create_task(bounded_run_request()))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate statistics
            successes = sum(1 for success, _ in results if success)
            case_results["success_rate"] = (successes / total_tasks) * 100 if total_tasks > 0 else 0
            case_results["throughput"] = successes / total_time if total_time > 0 else 0
            
            # Add detailed statistics for durations
            if case_results["durations"]:
                case_results.update(self._calculate_statistics(case_results["durations"]))
            
            # Store results for this test case
            self.results[results_key]["test_cases"][test_case["name"]] = case_results
        
        # Calculate aggregate statistics
        all_durations = []
        total_errors = 0
        total_success = 0
        
        for case_name, case_result in self.results[results_key]["test_cases"].items():
            all_durations.extend(case_result["durations"])
            total_errors += case_result["errors"]
            total_success += len(case_result["durations"])
        
        total_requests = total_success + total_errors
        self.results[results_key]["total_success"] = total_success
        self.results[results_key]["total_errors"] = total_errors
        self.results[results_key]["success_rate"] = (total_success / total_requests) * 100 if total_requests > 0 else 0
        
        if all_durations:
            self.results[results_key].update(self._calculate_statistics(all_durations))
            self.results[results_key]["total_throughput"] = total_success / sum(all_durations) if all_durations else 0
    
    def _calculate_statistics(self, durations: List[float]) -> Dict[str, float]:
        """
        Calculate statistical measures from a list of durations.
        
        Args:
            durations: List of duration measurements in seconds
            
        Returns:
            Dictionary containing statistical measures
        """
        if not durations:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "std_dev": 0
            }
        
        return {
            "min": min(durations),
            "max": max(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "p90": np.percentile(durations, 90),
            "p95": np.percentile(durations, 95),
            "p99": np.percentile(durations, 99),
            "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0
        }
    
    def generate_reports(self):
        """Generate benchmark reports and visualizations."""
        # Save raw results as JSON
        json_path = os.path.join(self.output_dir, f"benchmark_results_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate CSV report
        csv_path = os.path.join(self.output_dir, f"benchmark_results_{self.timestamp}.csv")
        self._generate_csv_report(csv_path)
        
        # Generate plots if enabled
        if self.config["generate_plots"]:
            self._generate_plots()
    
    def _generate_csv_report(self, csv_path: str):
        """
        Generate a CSV report of benchmark results.
        
        Args:
            csv_path: Path to save the CSV file
        """
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'component', 'test_case', 'concurrent_users', 
                'mean', 'median', 'min', 'max', 'p90', 'p95', 'p99',
                'success_rate', 'throughput'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write individual component results
            for component in ['llm_inference', 'mathematical_computation', 
                             'visualization_generation', 'database_operations',
                             'response_generation']:
                if component in self.results:
                    row = {
                        'component': component,
                        'test_case': 'all',
                        'concurrent_users': 1
                    }
                    row.update({k: v for k, v in self.results[component].items() 
                               if k in ['mean', 'median', 'min', 'max', 'p90', 'p95', 'p99']})
                    row['success_rate'] = 100  # Assuming all succeeded
                    row['throughput'] = 1 / row['mean'] if row['mean'] > 0 else 0
                    writer.writerow(row)
            
            # Write end-to-end results
            for key, result in self.results.items():
                if key.startswith('end_to_end_'):
                    concurrent_users = result['concurrent_users']
                    
                    # Write aggregate results
                    if 'mean' in result:
                        row = {
                            'component': 'end_to_end',
                            'test_case': 'all',
                            'concurrent_users': concurrent_users,
                            'mean': result.get('mean', 0),
                            'median': result.get('median', 0),
                            'min': result.get('min', 0),
                            'max': result.get('max', 0),
                            'p90': result.get('p90', 0),
                            'p95': result.get('p95', 0),
                            'p99': result.get('p99', 0),
                            'success_rate': result.get('success_rate', 0),
                            'throughput': result.get('total_throughput', 0)
                        }
                        writer.writerow(row)
                    
                    # Write per-test-case results
                    for test_case, case_result in result.get('test_cases', {}).items():
                        if 'mean' in case_result:
                            row = {
                                'component': 'end_to_end',
                                'test_case': test_case,
                                'concurrent_users': concurrent_users,
                                'mean': case_result.get('mean', 0),
                                'median': case_result.get('median', 0),
                                'min': case_result.get('min', 0),
                                'max': case_result.get('max', 0),
                                'p90': case_result.get('p90', 0),
                                'p95': case_result.get('p95', 0),
                                'p99': case_result.get('p99', 0),
                                'success_rate': case_result.get('success_rate', 0),
                                'throughput': case_result.get('throughput', 0)
                            }
                            writer.writerow(row)
    
    def _generate_plots(self):
        """Generate visualization plots of benchmark results."""
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, f"plots_{self.timestamp}")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Component comparison
        self._plot_component_comparison(plots_dir)
        
        # Plot 2: Latency by concurrency
        self._plot_latency_by_concurrency(plots_dir)
        
        # Plot 3: Throughput by concurrency
        self._plot_throughput_by_concurrency(plots_dir)
        
        # Plot 4: Success rate by concurrency
        self._plot_success_rate_by_concurrency(plots_dir)
        
        # Plot 5: Test case comparison
        self._plot_test_case_comparison(plots_dir)
    
    def _plot_component_comparison(self, plots_dir: str):
        """
        Plot performance comparison between components.
        
        Args:
            plots_dir: Directory to save the plot
        """
        components = ['llm_inference', 'mathematical_computation', 
                     'visualization_generation', 'database_operations',
                     'response_generation']
        
        means = []
        p95s = []
        labels = []
        
        for component in components:
            if component in self.results:
                means.append(self.results[component].get('mean', 0))
                p95s.append(self.results[component].get('p95', 0))
                labels.append(component.replace('_', ' ').title())
        
        if not means:  # Skip if no data
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, means, width, label='Mean', color='skyblue')
        ax.bar(x + width/2, p95s, width, label='95th Percentile', color='orange')
        
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Component Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'component_comparison.png'))
        plt.close()
    
    def _plot_latency_by_concurrency(self, plots_dir: str):
        """
        Plot latency metrics by concurrency level.
        
        Args:
            plots_dir: Directory to save the plot
        """
        concurrency_levels = []
        mean_latencies = []
        p95_latencies = []
        
        for key, result in sorted(self.results.items()):
            if key.startswith('end_to_end_'):
                if 'mean' in result and 'p95' in result:
                    concurrency_levels.append(result['concurrent_users'])
                    mean_latencies.append(result['mean'])
                    p95_latencies.append(result['p95'])
        
        if not concurrency_levels:  # Skip if no data
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(concurrency_levels, mean_latencies, 'o-', label='Mean Latency', color='blue')
        ax.plot(concurrency_levels, p95_latencies, 's-', label='95th Percentile Latency', color='red')
        
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Latency by Concurrency Level')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_by_concurrency.png'))
        plt.close()
    
    def _plot_throughput_by_concurrency(self, plots_dir: str):
        """
        Plot throughput by concurrency level.
        
        Args:
            plots_dir: Directory to save the plot
        """
        concurrency_levels = []
        throughputs = []
        
        for key, result in sorted(self.results.items()):
            if key.startswith('end_to_end_'):
                if 'total_throughput' in result:
                    concurrency_levels.append(result['concurrent_users'])
                    throughputs.append(result['total_throughput'])
        
        if not concurrency_levels:  # Skip if no data
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(concurrency_levels, throughputs, 'o-', label='Throughput', color='green')
        
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Throughput (requests/second)')
        ax.set_title('System Throughput by Concurrency Level')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'throughput_by_concurrency.png'))
        plt.close()
    
    def _plot_success_rate_by_concurrency(self, plots_dir: str):
        """
        Plot success rate by concurrency level.
        
        Args:
            plots_dir: Directory to save the plot
        """
        concurrency_levels = []
        success_rates = []
        
        for key, result in sorted(self.results.items()):
            if key.startswith('end_to_end_'):
                if 'success_rate' in result:
                    concurrency_levels.append(result['concurrent_users'])
                    success_rates.append(result['success_rate'])
        
        if not concurrency_levels:  # Skip if no data
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(concurrency_levels, success_rates, 'o-', label='Success Rate', color='purple')
        
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Concurrency Level')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 105)  # Set y-axis to 0-105% for clearer visualization
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'success_rate_by_concurrency.png'))
        plt.close()
    
    def _plot_test_case_comparison(self, plots_dir: str):
        """
        Plot performance comparison between test cases.
        
        Args:
            plots_dir: Directory to save the plot
        """
        # Get highest concurrency test for comparison
        highest_concurrency_key = None
        highest_concurrency = 0
        
        for key, result in self.results.items():
            if key.startswith('end_to_end_'):
                concurrency = result.get('concurrent_users', 0)
                if concurrency > highest_concurrency:
                    highest_concurrency = concurrency
                    highest_concurrency_key = key
        
        if not highest_concurrency_key:  # Skip if no data
            return
            
        test_cases = []
        mean_latencies = []
        
        for test_case, case_result in self.results[highest_concurrency_key].get('test_cases', {}).items():
            if 'mean' in case_result:
                test_cases.append(test_case)
                mean_latencies.append(case_result['mean'])
        
        if not test_cases:  # Skip if no data
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by latency for better visualization
        sorted_indices = np.argsort(mean_latencies)
        sorted_test_cases = [test_cases[i] for i in sorted_indices]
        sorted_latencies = [mean_latencies[i] for i in sorted_indices]
        
        y_pos = np.arange(len(sorted_test_cases))
        
        ax.barh(y_pos, sorted_latencies, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([tc.replace('_', ' ').title() for tc in sorted_test_cases])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Mean Response Time (seconds)')
        ax.set_title(f'Test Case Performance Comparison ({highest_concurrency} concurrent users)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'test_case_comparison.png'))
        plt.close()

async def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description='Performance Benchmark for Mathematical Multimodal LLM')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--api-url', type=str, help='Base URL for API')
    parser.add_argument('--iterations', type=int, help='Number of iterations per test')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
    
    # Override with command line arguments
    if args.api_url:
        config['api_base_url'] = args.api_url
    if args.iterations:
        config['iterations'] = args.iterations
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    benchmark = PerformanceBenchmark(config)
    await benchmark.run_benchmarks()

if __name__ == "__main__":
    asyncio.run(main())

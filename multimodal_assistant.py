#!/usr/bin/env python3
"""
Multimodal Mathematical Assistant
A unified interface that can detect user intent and handle both:
1. Mathematical problem solving
2. Function plotting and visualization

Uses Mistral 7B model with optimized settings for performance.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
import json
import gc
from llama_cpp import Llama
import io
import random
from typing import Dict, List, Union, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

class MultimodalAssistant:
    """
    A multimodal assistant that can handle text and mathematical queries,
    including generating plots and solving equations.
    """
    
    def __init__(self, model_dir="models/mistral-7b-instruct", 
                model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                context_length=4096, 
                num_threads=4,
                batch_size=512,
                gpu_layers=0):
        """
        Initialize the MultimodalAssistant with the specified model parameters.
        
        Args:
            model_dir: Directory containing the model
            model_file: Name of the model file
            context_length: Context window length for the model
            num_threads: Number of threads to use for computation
            batch_size: Batch size for inference
            gpu_layers: Number of layers to offload to GPU
        """
        self.model_dir = model_dir
        self.model_file = model_file
        self.model_path = os.path.join(model_dir, model_file)
        self.context_length = context_length
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.gpu_layers = gpu_layers
        self.model = None
        
        self.query_types = {
            "text": self.handle_text_query,
            "math": self.handle_math_query,
            "plot": self.handle_plot_query
        }
        
        # Log initialization details
        logger.info(f"Initializing MultimodalAssistant with:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Context length: {self.context_length}")
        logger.info(f"  Threads: {self.num_threads}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  GPU layers: {self.gpu_layers}")
    
    def load_model(self):
        """Load the LLM model."""
        logger.info(f"Loading model from {self.model_path}...")
        
        try:
            from llama_cpp import Llama
            
            start_time = time.time()
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_threads=self.num_threads,
                n_batch=self.batch_size,
                n_gpu_layers=self.gpu_layers,
                use_mlock=True,
                use_mmap=True,
                verbose=False
            )
            
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
            logger.info(f"Model has {self.model.n_ctx} context length and uses {self.model.n_threads} threads")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def generate_response(self, query, system_message=None, max_tokens=300, temperature=0.7, stream=False):
        """
        Generate a response from the model.
        
        Args:
            query: The user's query
            system_message: Optional system message to prepend
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            stream: Whether to stream the response
            
        Returns:
            The generated response
        """
        if not self.model:
            self.load_model()
        
        # Prepare the prompt
        if system_message:
            prompt = f"<s>[INST] {system_message}\n\n{query} [/INST]"
        else:
            prompt = f"<s>[INST] {query} [/INST]"
        
        # Generate the response
        start_time = time.time()
        logger.info(f"Generating response for query: {query[:50]}...")
        
        try:
            if stream:
                response_text = ""
                for token in self.model(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                ):
                    piece = token["choices"][0]["text"]
                    response_text += piece
                    # We can yield here if needed for streaming in the future
                
                full_response = response_text
            else:
                completion = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                full_response = completion["choices"][0]["text"]
            
            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f} seconds")
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def classify_query(self, query):
        """Determine the type of mathematical query."""
        if self.model is None:
            self.load_model()
        
        # Define a prompt to classify the query
        system_message = """<s>[INST] You are a mathematical assistant classifier.
Given a user query, determine what type of mathematical task they want to perform.
Return ONLY one of the following category labels, with no other text:
1. GRAPH - if they want to plot or visualize a function or equation
2. SOLVE - if they want to solve a math problem, including derivatives, integrals, etc.
3. EXPLAIN - if they want an explanation of a mathematical concept
4. UNKNOWN - if you cannot determine the type of query

For example:
- "Plot sin(x) from -π to π" → GRAPH
- "Find the derivative of 3x^2 + 2x - 5" → SOLVE
- "What is the chain rule in calculus?" → EXPLAIN
[/INST]
"""
        
        prompt = f"{system_message}\n\nUser query: {query}"
        
        response = self.model(
            prompt,
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.1,
            echo=False
        )
        
        result = response["choices"][0]["text"].strip().upper()
        
        # Extract just the category
        for category in ["GRAPH", "SOLVE", "EXPLAIN", "UNKNOWN"]:
            if category in result:
                logger.info(f"Query classified as: {category}")
                return category
        
        # Fallback classification based on keywords
        query_lower = query.lower()
        if any(word in query_lower for word in ["plot", "graph", "visualize", "draw", "chart"]):
            logger.info("Query classified as: GRAPH (keyword match)")
            return "GRAPH"
        elif any(word in query_lower for word in ["solve", "calculate", "compute", "find", "derivative", "integral"]):
            logger.info("Query classified as: SOLVE (keyword match)")
            return "SOLVE"
        elif any(word in query_lower for word in ["explain", "what is", "how does", "why", "describe"]):
            logger.info("Query classified as: EXPLAIN (keyword match)")
            return "EXPLAIN"
        
        logger.info("Query classified as: UNKNOWN (default)")
        return "UNKNOWN"
    
    def handle_graph_query(self, query):
        """Handle a query about plotting a function."""
        if self.model is None:
            self.load_model()
        
        # Interpret the mathematical function
        system_message = """<s>[INST] You are a mathematical function interpreter. 
I will give you a prompt about plotting a math function.
Your task is to:
1. Extract the exact mathematical function to plot
2. Determine appropriate x-axis range for plotting
3. Identify any special features or points of interest
4. Return ONLY a JSON-like response in this exact format:
{
  "function": "the exact function in Python syntax, e.g., 'sin(x)'",
  "x_min": minimum x value (number),
  "x_max": maximum x value (number),
  "title": "appropriate plot title",
  "features": ["list of special features or points"]
}
Only respond with valid Python math syntax that can be evaluated with NumPy.
For trigonometric functions, do not include 'np.' prefix, just use 'sin(x)', 'cos(x)', etc.
[/INST]
"""
        
        full_prompt = f"{system_message}\n\n{query}"
        
        # Generate interpretation
        response = self.model(
            full_prompt,
            max_tokens=200,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        result = response["choices"][0]["text"].strip()
        logger.info(f"Model response for graph interpretation: {result}")
        
        # Parse the response
        try:
            # Try multiple parsing methods
            # Method 1: Direct JSON parsing
            try:
                match = re.search(r'({[\s\S]*})', result)
                if match:
                    json_str = match.group(1)
                    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                    json_str = json_str.replace("'", '"')
                    data = json.loads(json_str)
                    return self.create_plot(data["function"], data["x_min"], data["x_max"], data["title"], data["features"])
            except:
                pass
                
            # Method 2: Extract components with regex
            function_match = re.search(r'"function"\s*:\s*"([^"]+)"', result)
            x_min_match = re.search(r'"x_min"\s*:\s*([-\d.]+)', result)
            x_max_match = re.search(r'"x_max"\s*:\s*([-\d.]+)', result)
            title_match = re.search(r'"title"\s*:\s*"([^"]+)"', result)
            
            if function_match:
                function = function_match.group(1)
                x_min = float(x_min_match.group(1)) if x_min_match else -10
                x_max = float(x_max_match.group(1)) if x_max_match else 10
                title = title_match.group(1) if title_match else "Function Plot"
                
                data = {
                    "function": function,
                    "x_min": x_min,
                    "x_max": x_max,
                    "title": title,
                    "features": ["Mathematical function"]
                }
                return self.create_plot(data["function"], data["x_min"], data["x_max"], data["title"], data["features"])
                
            # Method 3: Extract directly from the query
            query_lower = query.lower()
            
            # Check for basic functions
            basic_funcs = {
                'sin': ('sin(x)', -2*np.pi, 2*np.pi, 'Sine Function'),
                'cos': ('cos(x)', -2*np.pi, 2*np.pi, 'Cosine Function'),
                'tan': ('tan(x)', -np.pi/2 + 0.1, np.pi/2 - 0.1, 'Tangent Function'),
                'exp': ('exp(x)', -2, 2, 'Exponential Function'),
                'log': ('log(x)', 0.1, 5, 'Logarithmic Function'),
                'x^2': ('x**2', -5, 5, 'Quadratic Function'),
            }
            
            for key, (func, x_min, x_max, title) in basic_funcs.items():
                if key in query_lower:
                    data = {
                        "function": func,
                        "x_min": x_min,
                        "x_max": x_max,
                        "title": title,
                        "features": ["Automatic function"]
                    }
                    return self.create_plot(data["function"], data["x_min"], data["x_max"], data["title"], data["features"])
            
            # Extract custom expressions
            expr_match = re.search(r'(?:plot|graph|draw)\s+([\w\d\s\+\-\*\/\^\(\)]+)', query_lower)
            if expr_match:
                expr = expr_match.group(1).strip()
                expr = expr.replace('^', '**')
                data = {
                    "function": expr,
                    "x_min": -10,
                    "x_max": 10,
                    "title": f"Plot of {expr}",
                    "features": ["User-defined function"]
                }
                return self.create_plot(data["function"], data["x_min"], data["x_max"], data["title"], data["features"])
                
        except Exception as e:
            logger.error(f"Error processing graph query: {e}")
        
        # Fallback to a simple sine plot
        return self.create_plot("sin(x)", -2 * np.pi, 2 * np.pi, "Sine Function (Default)", ["Default function"])
    
    def create_plot(self, function_str, x_min, x_max, title="Function Plot", features=None):
        """Create a plot of the mathematical function."""
        plt.figure(figsize=(10, 6))
        
        # Generate x values
        x = np.linspace(x_min, x_max, 1000)
        
        try:
            # Safety check and function string cleanup
            cleaned_func_str = function_str.replace('np.np.', 'np.')
            
            # Define basic functions for direct evaluation
            basic_functions = {
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs
            }
            
            # Create a local namespace with NumPy and basic functions
            local_namespace = {'x': x, 'np': np}
            local_namespace.update(basic_functions)
            
            # Try to evaluate the function
            y = eval(cleaned_func_str, {"__builtins__": {}}, local_namespace)
            
            # Plot the function
            plt.plot(x, y, 'b-', linewidth=2)
            
            # Add grid, title, and labels
            plt.grid(True, alpha=0.3)
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('f(x)')
            
            # Highlight special features if provided
            if features:
                for feature in features:
                    if isinstance(feature, dict) and 'x' in feature and 'y' in feature:
                        plt.scatter([feature['x']], [feature['y']], color='red', s=50)
                        plt.annotate(feature.get('label', 'Point'), 
                                    (feature['x'], feature['y']), 
                                    textcoords="offset points", 
                                    xytext=(0,10), 
                                    ha='center')
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Close the plot to free memory
            plt.close()
            
            return buf
        except Exception as e:
            print(f"Error plotting function '{function_str}': {e}")
            # Fallback to a simple sin function plot
            x = np.linspace(-10, 10, 1000)
            y = np.sin(x)  # Using np.sin directly
            plt.plot(x, y, 'r--', linewidth=2)
            plt.title(f"Error plotting - Fallback plot\nError: {str(e)}")
            plt.grid(True, alpha=0.3)
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Close the plot to free memory
            plt.close()
            
            return buf
    
    def handle_solve_query(self, query):
        """Handle a query about solving a mathematical problem."""
        if self.model is None:
            self.load_model()
            
        # Format prompt for solving math problems
        system_message = """<s>[INST] You are a mathematics expert. 
Solve the following mathematical problem step by step.
Show all your work clearly, explaining each step.
If the problem involves calculus, show the relevant rules and formulas.
If it's an equation, show the solution process.
Keep your answer focused, precise, and mathematically rigorous.
[/INST]
"""
        
        prompt = f"{system_message}\n\n{query}"
        
        # Generate solution
        response = self.model(
            prompt,
            max_tokens=500,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        solution = response["choices"][0]["text"].strip()
        
        return {
            "type": "solution",
            "query": query,
            "solution": solution,
            "message": "Problem solved successfully."
        }
    
    def handle_explain_query(self, query):
        """Handle a query about explaining a mathematical concept."""
        if self.model is None:
            self.load_model()
            
        # Format prompt for explaining concepts
        system_message = """<s>[INST] You are a mathematics educator.
Explain the following mathematical concept clearly and concisely.
Include:
1. A clear definition
2. Key properties or rules
3. At least one simple example
4. Practical applications if relevant
Keep your explanation accessible to someone learning the topic.
[/INST]
"""
        
        prompt = f"{system_message}\n\n{query}"
        
        # Generate explanation
        response = self.model(
            prompt,
            max_tokens=500,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        explanation = response["choices"][0]["text"].strip()
        
        return {
            "type": "explanation",
            "query": query,
            "explanation": explanation,
            "message": "Concept explained successfully."
        }
    
    def process_query(self, query: str, query_type: str = "text", **kwargs):
        """
        Process a user query based on its type.
        
        Args:
            query: The user's query
            query_type: Type of query (text, math, plot)
            **kwargs: Additional arguments specific to the query type
            
        Returns:
            The response from the appropriate handler
        """
        if not self.model:
            logger.info("Model not loaded, loading now...")
            self.load_model()
            
        if query_type not in self.query_types:
            logger.warning(f"Unknown query type: {query_type}. Defaulting to text.")
            query_type = "text"
            
        logger.info(f"Processing {query_type} query: {query[:50]}...")
        
        try:
            start_time = time.time()
            handler = self.query_types[query_type]
            result = handler(query, **kwargs)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            
            # For plot queries, save the plot to a file if the result contains image data
            if query_type == "plot" and isinstance(result, tuple) and len(result) == 2:
                explanation, image_data = result
                if image_data:
                    # Create plots directory if it doesn't exist
                    os.makedirs("plots", exist_ok=True)
                    
                    # Generate a unique filename
                    filename = f"plots/plot_{int(time.time())}_{random.randint(1000, 9999)}.png"
                    
                    # Save the plot
                    with open(filename, "wb") as f:
                        f.write(image_data)
                    
                    return f"file://{os.path.abspath(filename)}"
                else:
                    return explanation
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def handle_text_query(self, query: str, **kwargs):
        """
        Handle a general text query.
        
        Args:
            query: The user's text query
            **kwargs: Additional arguments
            
        Returns:
            The model's response
        """
        max_tokens = kwargs.get("max_tokens", 300)
        temperature = kwargs.get("temperature", 0.7)
        
        return self.generate_response(query, max_tokens=max_tokens, temperature=temperature)
    
    def handle_math_query(self, query: str, **kwargs):
        """
        Handle a mathematical query.
        
        Args:
            query: The mathematical query
            **kwargs: Additional arguments
            
        Returns:
            The solution to the mathematical query
        """
        max_tokens = kwargs.get("max_tokens", 300)
        temperature = kwargs.get("temperature", 0.1)  # Lower temperature for math
        
        # Add a system instruction to enhance math capabilities
        system_message = """You are a mathematical assistant. Analyze the problem step-by-step.
        Show your work clearly. Use proper mathematical notation when needed.
        Verify your answer by checking your calculations."""
        
        return self.generate_response(
            query, 
            system_message=system_message,
            max_tokens=max_tokens, 
            temperature=temperature
        )
    
    def handle_plot_query(self, query: str, **kwargs):
        """
        Handle a query that requires plotting.
        
        Args:
            query: The query requiring visualization
            **kwargs: Additional arguments
            
        Returns:
            A tuple of (text_response, image_data) where image_data is the 
            binary representation of the generated plot
        """
        # First, get the model to understand what needs to be plotted
        system_message = """You are a data visualization expert. 
        Generate Python code to create the requested plot using matplotlib.
        The code should be complete and executable."""
        
        code_response = self.generate_response(
            query,
            system_message=system_message,
            max_tokens=500,
            temperature=0.2
        )
        
        # Extract Python code from the response
        code_blocks = []
        in_code_block = False
        code_lines = []
        
        for line in code_response.split('\n'):
            if line.strip() == '```python' or line.strip() == '```':
                if in_code_block:
                    code_blocks.append('\n'.join(code_lines))
                    code_lines = []
                in_code_block = not in_code_block
            elif in_code_block:
                code_lines.append(line)
        
        if code_lines:
            code_blocks.append('\n'.join(code_lines))
        
        if not code_blocks:
            return "I couldn't generate a proper visualization for this query.", None
        
        # Execute the code in a sandbox to generate the plot
        plot_data = None
        explanation = "Here's the visualization you requested:"
        
        try:
            # Create a buffer to capture the plot
            buf = io.BytesIO()
            
            # Execute the code in a controlled environment
            locals_dict = {}
            exec("import matplotlib.pyplot as plt\nimport numpy as np\n" + code_blocks[0], locals_dict)
            
            # Ensure the figure is saved to our buffer
            if 'plt.show()' in code_blocks[0]:
                exec("plt.savefig(buf, format='png')\nplt.close()", locals_dict)
            else:
                exec("plt.savefig(buf, format='png')\nplt.close()", locals_dict)
            
            buf.seek(0)
            plot_data = buf.getvalue()
            
        except Exception as e:
            explanation = f"I tried to create a visualization, but encountered an error: {str(e)}"
            logger.error(f"Error generating plot: {e}")
        
        return explanation, plot_data

def run_interactive_mode():
    """Run the assistant in interactive mode."""
    assistant = MultimodalAssistant()
    
    print("\n" + "="*70)
    print("  MULTIMODAL MATHEMATICS ASSISTANT")
    print("="*70)
    print("Ask me to solve problems, explain concepts, or plot functions.")
    print("Examples:")
    print("  - Plot sin(x) from -π to π")
    print("  - Find the derivative of 3x^2 + 2x - 5")
    print("  - Explain the chain rule in calculus")
    print("Type 'exit' to quit.")
    print("="*70 + "\n")
    
    while True:
        query = input("Your query: ")
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        assistant.process_query(query)
        print()

def main():
    """Main function for handling command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multimodal Mathematics Assistant"
    )
    
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Mathematical query (if not provided, runs in interactive mode)"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Process a single query from command line
        assistant = MultimodalAssistant()
        assistant.process_query(args.query)
    else:
        # Run in interactive mode
        run_interactive_mode()

if __name__ == "__main__":
    main() 
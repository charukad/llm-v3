#!/usr/bin/env python3
"""
Math Grapher - Generate mathematical plots from simple prompts.
Uses the Mistral model to interpret math expressions and matplotlib to create plots.
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
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def load_model():
    """Load the model with minimal settings for quick interpretation."""
    logger.info(f"Loading model from {MODEL_PATH}...")
    
    # Load model with minimal settings - we just need interpretation
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=8,      # Minimal GPU usage
        n_ctx=256,           # Small context window
        n_threads=1,         # Single thread
        n_batch=128,         # Small batch size
        use_mlock=True,      # Lock memory
        use_mmap=True,       # Memory mapping
        verbose=False        # No verbosity
    )
    
    return model

def interpret_math_prompt(model, prompt):
    """Use the model to interpret a mathematical prompt and extract the function."""
    system_message = """<[INST] You are a mathematical function interpreter. 
I will give you a simple prompt about plotting a math function.
Your task is to:
1. Extract the exact mathematical function to plot
2. Determine appropriate x-axis range for plotting
3. Identify any special features or points of interest
4. Return ONLY a JSON-like response in this exact format:
{
  "function": "the exact function in Python syntax, e.g., 'np.sin(x)'",
  "x_min": minimum x value (number),
  "x_max": maximum x value (number),
  "title": "appropriate plot title",
  "features": ["list of special features or points"]
}
Only respond with valid Python math syntax that can be evaluated with NumPy. [/INST]
"""
    
    full_prompt = f"{system_message}\n\n{prompt}"
    
    # Generate interpretation
    response = model(
        full_prompt,
        max_tokens=200,
        temperature=0.1,    # Low temperature for consistent results
        top_p=0.9,
        repeat_penalty=1.1,
        echo=False
    )
    
    result = response["choices"][0]["text"].strip()
    logger.info(f"Model response: {result}")
    
    # Try different extraction methods
    try:
        # Method 1: Direct JSON parsing
        try:
            # Find JSON-like structure with regex
            match = re.search(r'({[\s\S]*})', result)
            if match:
                json_str = match.group(1)
                # Fix common JSON formatting issues
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                
                # Parse JSON
                data = json.loads(json_str)
                return data
        except:
            pass
            
        # Method 2: Extract through regex matching each field
        function_match = re.search(r'"function"\s*:\s*"([^"]+)"', result)
        x_min_match = re.search(r'"x_min"\s*:\s*([-\d.]+)', result)
        x_max_match = re.search(r'"x_max"\s*:\s*([-\d.]+)', result)
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', result)
        features_match = re.search(r'"features"\s*:\s*\[(.*?)\]', result)
        
        if function_match:
            function = function_match.group(1)
            x_min = float(x_min_match.group(1)) if x_min_match else -2 * np.pi
            x_max = float(x_max_match.group(1)) if x_max_match else 2 * np.pi
            title = title_match.group(1) if title_match else "Function Plot"
            
            features = []
            if features_match:
                features_str = features_match.group(1)
                features = [f.strip().strip('"\'') for f in features_str.split(',') if f.strip()]
            
            return {
                "function": function,
                "x_min": x_min,
                "x_max": x_max,
                "title": title,
                "features": features or ["Mathematical function"]
            }
    except Exception as e:
        logger.error(f"Error parsing model output: {e}")
    
    # Method 3: Direct parsing of the prompt
    try:
        # Try to directly interpret common function patterns
        # Check for common functions like sin, cos, tan, etc.
        basic_funcs = {
            'sin': ('np.sin(x)', -2*np.pi, 2*np.pi, 'Sine Function'),
            'cos': ('np.cos(x)', -2*np.pi, 2*np.pi, 'Cosine Function'),
            'tan': ('np.tan(x)', -np.pi/2 + 0.1, np.pi/2 - 0.1, 'Tangent Function'),
            'exp': ('np.exp(x)', -2, 2, 'Exponential Function'),
            'log': ('np.log(x)', 0.1, 5, 'Logarithmic Function'),
            'sqrt': ('np.sqrt(x)', 0, 5, 'Square Root Function'),
            'x^2': ('x**2', -5, 5, 'Quadratic Function'),
            'x^3': ('x**3', -5, 5, 'Cubic Function'),
            'polynomial': ('x**2 - 2*x + 1', -5, 5, 'Polynomial Function')
        }
        
        prompt_lower = prompt.lower()
        for key, (func, x_min, x_max, title) in basic_funcs.items():
            if key in prompt_lower:
                return {
                    "function": func,
                    "x_min": x_min,
                    "x_max": x_max,
                    "title": title,
                    "features": ["Automatically interpreted function"]
                }
                
        # Check for custom expressions
        expr_match = re.search(r'(?:plot|graph|draw)\s+([\w\d\s\+\-\*\/\^\(\)]+)', prompt_lower)
        if expr_match:
            expr = expr_match.group(1).strip()
            # Convert to Python syntax
            expr = expr.replace('^', '**')
            return {
                "function": expr,
                "x_min": -10,
                "x_max": 10,
                "title": f"Plot of {expr}",
                "features": ["User-defined function"]
            }
    except:
        pass
        
    # Fallback to default
    logger.warning("Using default sine function as fallback")
    return {
        "function": "np.sin(x)",
        "x_min": -2 * np.pi,
        "x_max": 2 * np.pi,
        "title": "Sine Function",
        "features": ["Default fallback function"]
    }

def create_plot(data):
    """Create a plot based on the interpreted function."""
    # Generate x values
    x = np.linspace(data["x_min"], data["x_max"], 1000)
    
    try:
        # Process the function string for safe evaluation
        function_str = data["function"]
        # Replace ^ with ** for exponentiation if needed
        function_str = function_str.replace('^', '**')
        # Add np. prefix to common math functions if not present
        for func in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            pattern = r'(?<!\w)' + func + r'\('
            replacement = 'np.' + func + '('
            function_str = re.sub(pattern, replacement, function_str)
        
        # Safely evaluate the function
        y = eval(function_str)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        plt.title(data["title"])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add annotations for special features
        if "features" in data and data["features"]:
            feature_text = "\n".join(data["features"])
            plt.figtext(0.02, 0.02, f"Features: {feature_text}", wrap=True, fontsize=9)
        
        # Save the plot
        plot_file = "math_plot.png"
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Plot saved as {plot_file}")
        return plot_file
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        return None

def process_math_prompt(prompt):
    """Process a mathematical prompt and generate a plot."""
    start_time = time.time()
    
    # Load model
    model = load_model()
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Interpret the prompt
    logger.info(f"Interpreting prompt: {prompt}")
    interpretation_start = time.time()
    data = interpret_math_prompt(model, prompt)
    interpretation_time = time.time() - interpretation_start
    
    # Print the interpretation
    print("\nInterpreted function:")
    print(f"  Function: {data['function']}")
    print(f"  X range: [{data['x_min']}, {data['x_max']}]")
    print(f"  Title: {data['title']}")
    print(f"  Features: {', '.join(data['features'])}")
    
    # Create the plot
    logger.info("Creating plot...")
    plot_start = time.time()
    plot_file = create_plot(data)
    plot_time = time.time() - plot_start
    
    # Print timing information
    total_time = time.time() - start_time
    print(f"\nInterpreted prompt in {interpretation_time:.2f} seconds")
    print(f"Created plot in {plot_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    if plot_file:
        print(f"\nPlot saved as: {plot_file}")
        # Try to open the plot file
        try:
            if sys.platform == "darwin":  # macOS
                os.system(f"open {plot_file}")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {plot_file}")
            elif sys.platform == "linux":  # Linux
                os.system(f"xdg-open {plot_file}")
        except Exception as e:
            logger.error(f"Error opening plot file: {e}")
    else:
        print("Failed to create plot. Check logs for details.")

def main():
    """Main function for handling command line arguments."""
    parser = argparse.ArgumentParser(description="Generate mathematical plots from simple prompts")
    
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="plot sin(x)",
        help="Mathematical prompt to visualize, e.g., 'plot sin(x)'"
    )
    
    args = parser.parse_args()
    
    # Process the prompt
    process_math_prompt(args.prompt)

if __name__ == "__main__":
    main() 
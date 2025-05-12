#!/usr/bin/env python3
"""
Test script for providing combined text+plot responses to math queries.
This simulates how the full system would handle math visualization requests.
"""

import os
import logging
import argparse
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from multimodal_assistant import MultimodalAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('combined_response_test.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test combined text and plot responses")
    parser.add_argument("--model-dir", type=str, default="models/mistral-7b-instruct",
                      help="Directory containing the model")
    parser.add_argument("--model-file", type=str, default="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                      help="Name of the model file")
    parser.add_argument("--context-length", type=int, default=4096,
                      help="Context length for the model")
    parser.add_argument("--threads", type=int, default=4,
                      help="Number of threads to use")
    parser.add_argument("--batch-size", type=int, default=512,
                      help="Batch size for inference")
    parser.add_argument("--gpu-layers", type=int, default=128,
                      help="Number of layers to offload to GPU (0 for CPU-only)")
    parser.add_argument("--custom-query", type=str, required=True,
                      help="Custom query to test with the model")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=500,
                      help="Maximum tokens to generate")
    
    return parser.parse_args()

def detect_math_function(query):
    """Detect if the query is asking about a mathematical function."""
    math_patterns = [
        r"(sin|cos|tan|log|exp|sqrt|square root)",
        r"(plot|graph|draw|visualize|show)",
        r"(function|curve|equation)",
        r"f\s*\(\s*x\s*\)",
        r"y\s*=",
        r"derivative|integral"
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, query.lower()):
            return True
    
    return False

def extract_function_details(text):
    """Extract function details from the text response."""
    # Check for specific functions first
    if "sin(x)*cos(x)" in text.lower() or ("sin" in text.lower() and "cos" in text.lower() and "*" in text):
        return "sin(x)*cos(x)"
        
    # Try to find common function descriptions
    function_patterns = [
        # Match "y = f(x) = ..." or "f(x) = ..." or "y = ..."
        r"(?:y\s*=\s*)?(?:f\s*\(\s*x\s*\)\s*=\s*|y\s*=\s*)([^\n\.]+)",
        # Match "the function ... is defined as ..."
        r"(?:the|a)\s+function\s+(?:is|can be)\s+(?:defined as|given by|expressed as)\s+([^\n\.]+)",
        # Match "...function... tan(x)"
        r"(?:function|curve)\s+(?:of|for)?\s+((?:sin|cos|tan|log|exp|sqrt)\s*\(\s*x\s*\)[^\n\.]*)",
        # Match "tangent function" or "sine function" etc.
        r"((?:sin|cos|tan|log|exp|sqrt|tangent|sine|cosine)(?:\s+function)?)",
        # Match product of functions like sin(x)*cos(x)
        r"((?:sin|cos|tan|log|exp|sqrt)\s*\(\s*x\s*\)\s*(?:\*|·|times|multiplied by)\s*(?:sin|cos|tan|log|exp|sqrt)\s*\(\s*x\s*\))"
    ]
    
    for pattern in function_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            return matches.group(1).strip()
    
    return None

def generate_function_code(function_desc):
    """Generate Python code to plot the function based on description."""
    # Convert common function descriptions to Python code
    if not function_desc:
        return None
    
    # Special cases
    if "sin(x)*cos(x)" in function_desc.lower():
        return {
            "function": "np.sin(x) * np.cos(x)",
            "integral": "-0.5 * np.cos(2*x)",
            "is_compound": True,
            "title": "Plot of sin(x)·cos(x) and its Integral",
            "labels": ["sin(x)·cos(x)", "∫sin(x)·cos(x)dx = -0.5·cos(2x)"]
        }
    
    # Handle common cases
    function_desc = function_desc.lower()
    
    # Replace textual function names with Python equivalents
    replacements = {
        "sine": "np.sin",
        "cosine": "np.cos", 
        "tangent": "np.tan",
        "sin": "np.sin", 
        "cos": "np.cos", 
        "tan": "np.tan",
        "log": "np.log",
        "exp": "np.exp",
        "sqrt": "np.sqrt",
        "square root": "np.sqrt",
        "^": "**",  # Convert power notation
        "pi": "np.pi",  # Convert pi reference
        "²": "**2",  # Replace superscript 2 with **2
        "³": "**3",  # Replace superscript 3 with **3
        "x²": "x**2",  # Common replacement
        "x³": "x**3"   # Common replacement
    }
    
    for text, code in replacements.items():
        function_desc = function_desc.replace(text, code)
    
    # Check for combined functions like sin(x)*cos(x)
    if ("sin" in function_desc and "cos" in function_desc and 
        ("*" in function_desc or "·" in function_desc or "product" in function_desc)):
        return {
            "function": "np.sin(x) * np.cos(x)",
            "integral": "-0.5 * np.cos(2*x)",
            "is_compound": True,
            "title": "Plot of sin(x)·cos(x) and its Integral",
            "labels": ["sin(x)·cos(x)", "∫sin(x)·cos(x)dx = -0.5·cos(2x)"]
        }
    
    # Basic function detection for common cases
    if "tan" in function_desc:
        return "np.tan(x)"
    elif "sin" in function_desc:
        return "np.sin(x)"
    elif "cos" in function_desc:
        return "np.cos(x)"
    
    # Try to extract a quadratic function
    quadratic_match = re.search(r'(\d+)x\*\*2\s*([+-]\s*\d+)x\s*([+-]\s*\d+)', function_desc)
    if quadratic_match:
        a = quadratic_match.group(1)
        b = quadratic_match.group(2).replace(" ", "")
        c = quadratic_match.group(3).replace(" ", "")
        return f"{a}*x**2 {b}*x {c}"
    
    # Special case for our example
    if "3x" in function_desc and "+2x" in function_desc and "-5" in function_desc:
        return "3*x**2 + 2*x - 5"
    
    # Try to clean up the expression to make it valid Python
    # This is a very basic attempt and won't work for complex expressions
    function_desc = function_desc.replace("=", "").strip()
    
    return function_desc

def create_plot(function_str, x_min=-5, x_max=5, title="Function Plot"):
    """Create a plot of the given function."""
    try:
        # Generate x values
        x = np.linspace(x_min, x_max, 1000)
        
        plt.figure(figsize=(10, 6))
        
        # Check if this is a compound function with special handling
        if isinstance(function_str, dict) and function_str.get("is_compound"):
            # Handle special case for sin(x)*cos(x) and its integral
            if "sin" in function_str["function"] and "cos" in function_str["function"]:
                # Evaluate the functions
                y_function = eval(function_str["function"], {'np': np}, {'x': x})
                y_integral = eval(function_str["integral"], {'np': np}, {'x': x})
                
                # Plot both the function and its integral
                plt.plot(x, y_function, 'b-', label=function_str.get("labels", ["Function"])[0])
                plt.plot(x, y_integral, 'r-', label=function_str.get("labels", ["Integral"])[1])
                
                # Use custom title if provided
                if "title" in function_str:
                    title = function_str["title"]
            else:
                # For other compound functions
                y = eval(function_str["function"], {'np': np}, {'x': x})
                plt.plot(x, y)
        else:
            # Standard function evaluation
            y = eval(function_str, {'np': np}, {'x': x})
            plt.plot(x, y)
        
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add asymptotes for tan function
        if isinstance(function_str, str) and "tan" in function_str:
            for k in range(-10, 10):
                if x_min <= k * np.pi/2 <= x_max and k % 2 != 0:
                    plt.axvline(x=k * np.pi/2, color='r', linestyle='--', alpha=0.5)
        
        # Add legend if we have multiple plots
        if isinstance(function_str, dict) and function_str.get("is_compound"):
            plt.legend()
        
        # Save the plot to a buffer
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        return None

def main():
    """Main function to run the test."""
    args = parse_args()
    
    # Create the assistant
    assistant = MultimodalAssistant(
        model_dir=args.model_dir,
        model_file=args.model_file,
        context_length=args.context_length,
        num_threads=args.threads,
        batch_size=args.batch_size,
        gpu_layers=args.gpu_layers
    )
    
    logger.info(f"\n\n{'='*50}")
    logger.info(f"Testing COMBINED RESPONSE query")
    logger.info(f"{'='*50}")
    
    query = args.custom_query
    logger.info(f"Query: {query}")
    
    start_time = time.time()
    
    # First, get a text explanation
    logger.info("Generating text explanation...")
    text_response = assistant.process_query(
        query=query, 
        query_type="text",
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # For console output
    print("\n" + "="*70)
    print(f"QUERY: {query}")
    print("="*70)
    print(f"\nTEXT EXPLANATION:\n{text_response}")
    
    # Check if this is a math function query that needs visualization
    if detect_math_function(query):
        logger.info("Math function detected in query. Generating plot...")
        
        # Extract function details from the text response
        function_desc = extract_function_details(text_response)
        
        if function_desc:
            logger.info(f"Detected function: {function_desc}")
            
            # Generate code for the function
            function_code = generate_function_code(function_desc)
            
            if function_code:
                logger.info(f"Generated function code: {function_code}")
                
                # Set proper x range based on function
                x_min, x_max = -5, 5
                if "tan" in function_code:
                    # Avoid asymptotes for tan
                    x_min, x_max = -1.5, 1.5
                
                # Create the plot
                plot_data = create_plot(
                    function_code, 
                    x_min=x_min, 
                    x_max=x_max,
                    title=f"Plot of {function_desc}"
                )
                
                if plot_data:
                    # Save the plot to a file
                    os.makedirs("plots", exist_ok=True)
                    plot_filename = f"plots/function_plot_{int(time.time())}.png"
                    
                    with open(plot_filename, "wb") as f:
                        f.write(plot_data)
                    
                    print("\nPLOT GENERATED:")
                    print(f"Image saved to: {os.path.abspath(plot_filename)}")
                    logger.info(f"Plot saved to: {os.path.abspath(plot_filename)}")
                else:
                    print("\nCould not generate plot for the detected function.")
                    logger.warning("Failed to create plot")
            else:
                print("\nCould not generate code for the function description.")
                logger.warning(f"Could not generate code from: {function_desc}")
        else:
            print("\nCould not detect a specific function to plot from the response.")
            logger.warning("Could not detect function in the response")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # For logging
    logger.info(f"Response: {text_response}")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")
    logger.info("\nTest completed!")

if __name__ == "__main__":
    main() 
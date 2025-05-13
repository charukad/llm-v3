#!/usr/bin/env python3
"""
Direct test for NLP Visualization workflow.

This script demonstrates the full workflow:
1. Process natural language with LLM to extract visualization parameters
2. Generate visualization using the extracted parameters
3. Return the visualization result
"""

import sys
import json
import logging
import uuid
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the necessary components
try:
    from core.agent.llm_agent import CoreLLMAgent
    from visualization.agent.viz_agent import VisualizationAgent
    from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent
    import numpy as np
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)

def clean_json_string(json_str: str) -> str:
    """
    Clean and fix common JSON string issues.
    
    Args:
        json_str: The JSON string to clean
        
    Returns:
        Cleaned JSON string
    """
    # Remove comments
    json_str = re.sub(r'//.*?\n', '\n', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    
    # Fix common issues with expression parameters
    # Replace "z = expr" with just "expr"
    json_str = re.sub(r'"expression"\s*:\s*"z\s*=\s*([^"]+)"', r'"expression": "\1"', json_str)
    json_str = re.sub(r'"expression"\s*:\s*"f\(x,y\)\s*=\s*([^"]+)"', r'"expression": "\1"', json_str)
    # Also fix "y = expr" for function_2d
    json_str = re.sub(r'"expression"\s*:\s*"y\s*=\s*([^"]+)"', r'"expression": "\1"', json_str)
    json_str = re.sub(r'"expression"\s*:\s*"f\(x\)\s*=\s*([^"]+)"', r'"expression": "\1"', json_str)
    
    # Fix math operations in JSON
    json_str = re.sub(r'(\d+)\s*\*\s*(\d+|\w+\.\w+)', lambda m: str(float(m.group(1)) * float(m.group(2)) if m.group(2).replace('.', '', 1).isdigit() else f"{m.group(1)} * {m.group(2)}"), json_str)
    
    # Remove trailing commas before closing brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Replace JavaScript-specific math notations
    json_str = json_str.replace('Math.PI', '3.14159')
    
    return json_str

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to native Python types to ensure JSON serialization works.
    
    Args:
        obj: Object that might contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def process_nlp_visualization(prompt: str) -> Dict[str, Any]:
    """
    Process natural language prompt to generate visualization.
    
    Args:
        prompt: Natural language description of the visualization
        
    Returns:
        Dictionary with visualization results
    """
    logger.info(f"Processing prompt: {prompt}")
    
    # Initialize agents
    llm_agent = CoreLLMAgent()
    viz_agent = VisualizationAgent({"storage_dir": "visualizations", "use_database": True})
    advanced_viz_agent = AdvancedVisualizationAgent({"storage_dir": "visualizations", "use_database": True})
    
    # Step 1: Extract visualization parameters using LLM
    extraction_prompt = f"""
You are an AI specialized in extracting visualization parameters from natural language descriptions.
Extract the visualization type and all necessary parameters from the following text:

"{prompt}"

Format your response as a valid JSON object with the following structure:
{{
  "visualization_type": "one of [function_2d, functions_2d, function_3d, parametric_3d, histogram, scatter]",
  "parameters": {{
    // All necessary parameters for the chosen visualization type
    // For scatter plots: x_data, y_data, title, etc.
    // For function plots: expression, x_range, etc.
    // For histograms: data, bins, etc.
  }}
}}

IMPORTANT: For function_3d type, don't include 'z = ' or 'f(x,y) = ' in the expression value, just provide the math formula itself.
Example: For "z = sin(x)*cos(y)" just use "sin(x)*cos(y)" as the expression value.

Be accurate and ensure all numeric data is properly formatted as numbers, not strings.
Lists should be proper JSON arrays.
"""
    
    logger.info("Sending prompt to LLM for parameter extraction")
    llm_response = llm_agent.generate_response(extraction_prompt)
    
    if not llm_response.get("success", False):
        logger.error(f"LLM extraction failed: {llm_response.get('error', 'Unknown error')}")
        return {
            "success": False,
            "error": f"LLM extraction failed: {llm_response.get('error', 'Unknown error')}"
        }
    
    # Extract JSON from LLM response
    response_text = llm_response.get("response", "")
    logger.info(f"LLM response: {response_text[:200]}...")
    
    try:
        # Extract JSON from the response text
        import re
        json_matches = re.findall(r'({[\s\S]*?})', response_text)
        
        extracted_data = None
        
        if json_matches:
            # Take the longest match, which is likely the full JSON
            json_str = max(json_matches, key=len)
            logger.debug(f"Extracted JSON string: {json_str}")
            
            # Clean the JSON string
            cleaned_json = clean_json_string(json_str)
            logger.debug(f"Cleaned JSON string: {cleaned_json}")
            
            try:
                extracted_data = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error after cleaning: {e}")
                logger.error(f"Problematic JSON: {cleaned_json}")
        
        if not extracted_data:
            # Try to extract just the JSON part using regex
            pattern = r'{\s*"visualization_type"\s*:.*?}'
            matches = re.search(pattern, response_text, re.DOTALL)
            if matches:
                json_str = matches.group(0)
                logger.debug(f"Extracted with regex: {json_str}")
                
                # Clean the JSON string
                cleaned_json = clean_json_string(json_str)
                logger.debug(f"Cleaned JSON string: {cleaned_json}")
                
                try:
                    extracted_data = json.loads(cleaned_json)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error after second attempt: {e}")
        
        if not extracted_data:
            # Try to manually parse the response
            vis_type_match = re.search(r'"visualization_type"\s*:\s*"([^"]+)"', response_text)
            if vis_type_match:
                visualization_type = vis_type_match.group(1)
                
                # Extract parameters based on visualization type
                if visualization_type == "function_3d":
                    expr_match = re.search(r'"expression"\s*:\s*"([^"]+)"', response_text)
                    expression = expr_match.group(1) if expr_match else "sin(x)*cos(y)"
                    
                    # Clean the expression (remove z = or f(x,y) = if present)
                    expression = re.sub(r'^z\s*=\s*', '', expression)
                    expression = re.sub(r'^f\s*\(\s*x\s*,\s*y\s*\)\s*=\s*', '', expression)
                    
                    extracted_data = {
                        "visualization_type": "function_3d",
                        "parameters": {
                            "expression": expression,
                            "x_range": [-10, 10],
                            "y_range": [-10, 10],
                            "title": "3D Surface Plot",
                        }
                    }
                    
                elif visualization_type == "function_2d":
                    expr_match = re.search(r'"expression"\s*:\s*"([^"]+)"', response_text)
                    expression = expr_match.group(1) if expr_match else "sin(x)"
                    
                    # Clean the expression (remove y = or f(x) = if present)
                    expression = re.sub(r'^y\s*=\s*', '', expression)
                    expression = re.sub(r'^f\s*\(\s*x\s*\)\s*=\s*', '', expression)
                    
                    extracted_data = {
                        "visualization_type": "function_2d",
                        "parameters": {
                            "expression": expression,
                            "x_range": [-10, 10],
                            "title": "Function Plot",
                        }
                    }
        
        # If still no data, create a default based on the prompt
        if not extracted_data:
            logger.warning("Couldn't parse JSON from LLM response, creating default parameters")
            
            # Check for multiple functions on the same graph
            if ("functions" in prompt.lower() or "plots" in prompt.lower() or "same graph" in prompt.lower()) and any(func in prompt.lower() for func in ["sin", "cos", "tan", "log", "exp"]):
                # Try to extract the functions from the prompt
                functions = []
                labels = []
                
                # Common trigonometric functions
                for func in ["sin(x)", "cos(x)", "tan(x)", "log(x)", "exp(x)", "x^2", "x**2", "sqrt(x)", "1/x"]:
                    if func in prompt.lower():
                        functions.append(func.replace("^", "**"))
                        labels.append(func)
                
                # Check for sum or combinations
                if "sum" in prompt.lower() or "+" in prompt:
                    # Try to extract combined expressions
                    combined_expr = None
                    for expr in re.findall(r'(?:sum|their sum)\s+([^,\.]+)', prompt):
                        combined_expr = expr.strip()
                        break
                    
                    if not combined_expr and len(functions) >= 2:
                        combined_expr = "+".join(functions)
                    
                    if combined_expr:
                        functions.append(combined_expr.replace("^", "**"))
                        labels.append("Sum")
                
                # If no functions found, use defaults
                if not functions:
                    functions = ["sin(x)", "cos(x)", "sin(x)+cos(x)"]
                    labels = ["sin(x)", "cos(x)", "Sum"]
                
                # Try to extract x range
                x_range = [-3.14159, 3.14159]  # Default to -π to π
                
                range_match = re.search(r'from\s+(-?\s*\d*\.?\d*π?)\s+to\s+(-?\s*\d*\.?\d*π?)', prompt)
                if range_match:
                    x_min = range_match.group(1).replace('π', str(3.14159)).replace('pi', str(3.14159))
                    x_max = range_match.group(2).replace('π', str(3.14159)).replace('pi', str(3.14159))
                    
                    # Handle cases like -π (just the symbol)
                    if x_min.strip() == '-':
                        x_min = '-1'
                    if x_max.strip() == '':
                        x_max = '1'
                        
                    try:
                        x_range = [float(eval(x_min)), float(eval(x_max))]
                    except:
                        # Fall back to default range
                        pass
                
                # Create visualization parameters
                extracted_data = {
                    "visualization_type": "functions_2d",
                    "parameters": {
                        "expressions": functions,
                        "labels": labels,
                        "x_range": x_range,
                        "title": "Multiple Functions Plot",
                        "x_label": "x",
                        "y_label": "y"
                    }
                }
                
                # Debug printing
                logger.debug(f"Extracted functions: {functions}")
                logger.debug(f"Extracted labels: {labels}")
                
                # Manually fix expressions format
                if extracted_data["visualization_type"] == "functions_2d":
                    # Ensure expressions don't contain "from -π to π" etc.
                    fixed_expressions = []
                    for expr in extracted_data["parameters"]["expressions"]:
                        # Remove any "from ... to ..." text
                        expr = re.sub(r'\s+from\s+.*$', '', expr)
                        fixed_expressions.append(expr)
                    
                    extracted_data["parameters"]["expressions"] = fixed_expressions
                    logger.debug(f"Fixed expressions: {fixed_expressions}")
            
            # Check if this is a histogram request
            elif "histogram" in prompt.lower() and ("distribution" in prompt.lower() or "random" in prompt.lower()):
                # Extract mean and standard deviation if present
                mean = 0
                std_dev = 1
                num_samples = 1000
                
                mean_match = re.search(r'mean\s+(\d+\.?\d*)', prompt.lower())
                if mean_match:
                    mean = float(mean_match.group(1))
                
                std_match = re.search(r'(?:standard deviation|std|deviation)\s+(\d+\.?\d*)', prompt.lower())
                if std_match:
                    std_dev = float(std_match.group(1))
                
                samples_match = re.search(r'(\d+)\s+(?:random\s+)?(?:numbers|data\s+points|samples)', prompt.lower())
                if samples_match:
                    num_samples = int(samples_match.group(1))
                
                # Generate random data
                np.random.seed(42)  # For reproducibility
                data = np.random.normal(mean, std_dev, num_samples).tolist()
                
                extracted_data = {
                    "visualization_type": "histogram",
                    "parameters": {
                        "data": data,
                        "bins": 30,
                        "title": f"Normal Distribution (μ={mean}, σ={std_dev})",
                        "x_label": "Value",
                        "y_label": "Frequency",
                        "show_kde": True
                    }
                }
            elif "3d" in prompt.lower() and "surface" in prompt.lower():
                # Try to extract expression with regex
                expr_match = re.search(r'z\s*=\s*([^.]+)', prompt)
                expression = expr_match.group(1).strip() if expr_match else "sin(x)*cos(y)"
                
                extracted_data = {
                    "visualization_type": "function_3d",
                    "parameters": {
                        "expression": expression,
                        "x_range": [-10, 10],
                        "y_range": [-10, 10],
                        "title": "3D Surface Plot from Natural Language",
                        "x_label": "x",
                        "y_label": "y",
                        "z_label": "z"
                    }
                }
            elif "sin" in prompt.lower() or "cos" in prompt.lower() or "tan" in prompt.lower():
                expr_match = re.search(r'f\s*\(\s*x\s*\)\s*=\s*([^.]+)', prompt)
                expression = expr_match.group(1).strip() if expr_match else "sin(x)"
                
                extracted_data = {
                    "visualization_type": "function_2d",
                    "parameters": {
                        "expression": expression,
                        "x_range": [-10, 10],
                        "title": "Function Plot from Natural Language",
                        "x_label": "x",
                        "y_label": "f(x)"
                    }
                }
            else:
                # Default to a histogram
                extracted_data = {
                    "visualization_type": "histogram",
                    "parameters": {
                        "data": list(range(1, 101)),  # Simple data 1-100
                        "bins": 20,
                        "title": "Histogram from Natural Language",
                        "x_label": "Value",
                        "y_label": "Frequency"
                    }
                }
            
        # Validate the extracted data
        if "visualization_type" not in extracted_data:
            raise ValueError("Missing visualization_type in extracted data")
        if "parameters" not in extracted_data:
            raise ValueError("Missing parameters in extracted data")
        
        # Step 2: Generate the visualization
        visualization_type = extracted_data["visualization_type"]
        parameters = extracted_data["parameters"]
        
        logger.info(f"Extracted visualization type: {visualization_type}")
        logger.info(f"Extracted parameters: {parameters}")
        
        # Generate a unique filename if not provided
        if "filename" not in parameters:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            parameters["filename"] = f"nlp_{visualization_type}_{timestamp}_{unique_id}.png"
        
        # Create message for visualization agent
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "sender": "nlp_visualization_test",
                "recipient": "visualization_agent", 
                "timestamp": datetime.now().isoformat(),
                "message_type": "visualization_request"
            },
            "body": {
                "visualization_type": visualization_type,
                "parameters": parameters
            }
        }
        
        # Determine which agent to use
        agent = viz_agent
        advanced_types = advanced_viz_agent.get_capabilities().get("advanced_features", [])
        
        if visualization_type in advanced_types:
            agent = advanced_viz_agent
        
        # Generate the visualization
        logger.info(f"Generating {visualization_type} visualization")
        result = agent.process_message(message)
        
        # Add LLM analysis to the result
        result["llm_analysis"] = response_text[:500]
        
        # Convert NumPy types to Python native types for JSON serialization
        result = convert_numpy_types(result)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error processing visualization: {e}")
        return {
            "success": False,
            "error": f"Error processing visualization: {str(e)}"
        }

def main():
    """Main function to run the test."""
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = input("Enter a visualization prompt: ")
    
    result = process_nlp_visualization(prompt)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))
    
    if result.get("success", False) and "file_path" in result:
        print(f"\nVisualization saved to: {result['file_path']}")

if __name__ == "__main__":
    main() 
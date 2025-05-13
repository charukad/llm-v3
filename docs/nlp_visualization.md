# Natural Language Visualization Guide

This guide explains how to use the Natural Language Visualization API endpoint to generate visualizations from natural language descriptions.

## Overview

The NLP Visualization endpoint allows you to generate visualizations by describing them in natural language. The system uses an LLM to interpret your description and extract the necessary parameters, then uses the appropriate visualization functions to create the plot.

## Endpoint

- **URL**: `/nlp-visualization`
- **Method**: `POST`
- **Content-Type**: `application/json`

## Request Format

```json
{
  "prompt": "Your natural language description of the visualization",
  "conversation_id": "optional_conversation_id",
  "user_id": "optional_user_id"
}
```

### Parameters

- `prompt` (required): A natural language description of the visualization you want to create
- `conversation_id` (optional): An identifier for the conversation this visualization is a part of
- `user_id` (optional): An identifier for the user requesting the visualization

## Response Format

```json
{
  "success": true,
  "visualization_type": "function_2d",
  "file_path": "/path/to/visualization.png",
  "base64_image": "base64_encoded_image_data",
  "llm_analysis": "LLM's interpretation of the prompt"
}
```

### Response Fields

- `success`: Boolean indicating whether the visualization was created successfully
- `visualization_type`: The type of visualization that was created
- `file_path`: The path to the saved visualization file (if saved to disk)
- `base64_image`: Base64-encoded image data (if returned as base64)
- `error`: Error message (if `success` is `false`)
- `llm_analysis`: The LLM's interpretation of your prompt (useful for debugging)

## Example Usage

### Example 1: Simple Scatter Plot

**Request:**
```json
{
  "prompt": "Create a scatter plot of 10 points with x values from 1 to 10 and y values: 2, 3, 5, 4, 6, 7, 8, 7, 9, 10. Add a regression line."
}
```

**Response:**
```json
{
  "success": true,
  "visualization_type": "scatter",
  "file_path": "visualizations/nlp_scatter_20230513_123456_abcd1234.png",
  "llm_analysis": "..."
}
```

### Example 2: Mathematical Function Plot

**Request:**
```json
{
  "prompt": "Plot the function f(x) = sin(x) + cos(x) in the range of x from -2π to 2π."
}
```

**Response:**
```json
{
  "success": true,
  "visualization_type": "function_2d",
  "file_path": "visualizations/nlp_function_2d_20230513_123456_abcd1234.png",
  "llm_analysis": "..."
}
```

## Supported Visualization Types

The system supports various visualization types, including:

1. `function_2d`: 2D plot of a mathematical function
2. `functions_2d`: Multiple 2D functions on one plot
3. `function_3d`: 3D surface plot of a function
4. `parametric_3d`: 3D parametric curve
5. `histogram`: Histogram of a dataset
6. `scatter`: Scatter plot with optional regression line

## Advanced Features

The system also supports more advanced visualization features, including:

1. `derivatives`: Derivative plots
2. `integrals`: Integral plots
3. `taylor_series`: Taylor series expansions
4. `critical_points`: Critical points visualization
5. `vector_fields`: Vector field plots

## Testing the API

You can test the API using the provided test scripts:

```bash
# Test with the default scatter plot example
python test_nlp_visualization.py

# Test with a custom prompt
python test_nlp_visualization.py --prompt "Plot the function f(x) = x^2 - 3x + 2 from x=-2 to x=5"

# Test various advanced visualization types
python test_advanced_nlp_viz.py --type function
python test_advanced_nlp_viz.py --type multiple
python test_advanced_nlp_viz.py --type 3d
python test_advanced_nlp_viz.py --type parametric
python test_advanced_nlp_viz.py --type scatter
python test_advanced_nlp_viz.py --type histogram
```

## Troubleshooting

If you encounter issues:

1. Check the `llm_analysis` field in the response to see how the LLM interpreted your prompt
2. Make sure your prompt clearly specifies all necessary parameters
3. Try providing more explicit instructions about the visualization type you want
4. If generating complex mathematical functions, use standard notation (e.g., sin(x), x^2, etc.) 
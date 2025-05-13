# Enhanced Visualization System

This system provides advanced visualization capabilities through natural language descriptions. The SuperVisualizationAgent extends the standard visualization capabilities with many additional plot types and enhanced features.

## Visualization Types

The SuperVisualizationAgent supports the following visualization types:

### Basic Visualizations
- **function_2d**: Plot a 2D function (e.g., `f(x) = sin(x)`)
- **functions_2d**: Plot multiple 2D functions on the same axes
- **function_3d**: Plot a 3D surface (e.g., `f(x,y) = sin(x)*cos(y)`)
- **parametric_3d**: Plot a 3D parametric curve
- **histogram**: Plot a histogram of data
- **scatter**: Create a scatter plot with optional regression line

### Statistical Visualizations
- **boxplot**: Compare distributions with box plots
- **violin**: Show distribution shapes with violin plots
- **bar**: Create bar charts for categorical data
- **pie**: Create pie charts for proportional data
- **heatmap**: Create heatmaps for visualizing matrices
- **correlation_matrix**: Visualize correlations between variables

### Advanced Mathematical Visualizations
- **derivative**: Plot a function and its derivative
- **critical_points**: Plot a function with its critical points highlighted
- **integral**: Plot a function with its integral region highlighted
- **taylor_series**: Plot a function with its Taylor series approximations
- **vector_field**: Plot a 2D vector field
- **contour**: Create contour plots of 2D functions
- **complex_function**: Visualize complex functions with domain coloring
- **slope_field**: Show direction fields for ODEs
- **phase_portrait**: Show phase portraits for dynamic systems

### Time Series and Specialized Visualizations
- **time_series**: Plot time series data with multiple series support

## NLP Visualization Endpoint

The system provides an API endpoint that accepts natural language descriptions of visualizations and returns the generated visualization:

```
POST /nlp-visualization
```

### Request Body
```json
{
  "prompt": "Plot the function f(x) = sin(x) * exp(-x/5) from -10 to 10"
}
```

### Response
```json
{
  "success": true,
  "visualization_type": "function_2d",
  "file_path": "visualizations/function_2d_20250513_180412_a1b2c3d4.png",
  "base64_image": null,
  "error": null,
  "llm_analysis": {"analysis": "Extracted parameters with enhanced NLP processing"}
}
```

## Example Prompts

Here are some example prompts that you can use with the NLP visualization endpoint:

- "Plot the function f(x) = sin(x) * cos(2*x) from -π to π"
- "Create a 3D surface plot of z = sin(sqrt(x^2 + y^2))"
- "Generate a scatter plot with these points: (1,3), (2,5), (3,4), (4,7), (5,8), (6,10) and show the trend line"
- "Create a boxplot comparing three groups: [1,2,3,4,5,6], [4,5,6,7,8], [2,3,4,5]"
- "Create a pie chart showing market share: 30% for Apple, 25% for Samsung, 15% for Xiaomi and 30% for Others"
- "Generate a contour plot of the function f(x,y) = x^2 + y^2"
- "Plot the slope field for the differential equation dy/dx = y"
- "Create a correlation matrix visualization for 5 variables"

## Testing

You can test the NLP visualization endpoint using the provided test scripts:

1. `test_nlp_visualization.py` - Tests a single visualization prompt
2. `test_super_viz.py` - Tests multiple visualization types with various prompts

Example usage:
```
python test_nlp_visualization.py --prompt "Plot the function f(x) = sin(x)" --url http://localhost:8000
```

```
python test_super_viz.py --url http://localhost:8000
```

## Implementation

The visualization system includes:

1. `SuperVisualizationAgent` - Enhanced visualization agent with advanced capabilities
2. Pattern matching algorithms for extracting visualization parameters directly from text
3. LLM integration for complex parameter extraction
4. Advanced parameter handling and validation
5. Robust error handling and fallbacks

When you enter a natural language prompt, the system:
1. First attempts direct pattern matching to extract parameters
2. If that fails, it uses the LLM for parameter extraction
3. Generates the visualization using the appropriate technique
4. Returns either a file path to the saved image or a base64-encoded image

## Dependencies

- Python 3.8+
- NumPy
- SymPy
- Matplotlib
- Seaborn
- FastAPI (for the API endpoint)

## Registration

To register the SuperVisualizationAgent, run:
```
python register_super_viz_agent.py
```

This will create and register the agent with the system, making all the enhanced visualization capabilities available. 
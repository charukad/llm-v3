# Debugging Visualization Detection

## Overview

The NLP visualization system includes a special debug endpoint that helps troubleshoot visualization detection issues. This document explains how to use this endpoint effectively.

## The Debug Endpoint

The debug endpoint is available at:

```
POST /api/nlp-visualization/debug
```

This endpoint accepts the same input as the regular visualization endpoint but provides detailed diagnostic information rather than generating a visualization.

## Request Format

```json
{
  "prompt": "Your natural language visualization request"
}
```

## Response Format

The debug response includes:

```json
{
  "prompt": "Original prompt",
  "detection_method": "Pattern matching or LLM extraction",
  "visualization_type": "Detected visualization type",
  "pattern_match_detected": true/false,
  "pattern_match_result": { /* Details about pattern matching result */ },
  "pattern_match_logs": [ /* Logs from pattern matching process */ ],
  "llm_detected": true/false,
  "llm_result": { /* Details about LLM extraction result */ },
  "llm_logs": [ /* Logs from LLM extraction process */ ],
  "supported_types": [ /* List of supported visualization types */ ],
  "recommendation": "Recommended approach"
}
```

## Using the Debug Endpoint

### When to Use It

The debug endpoint is particularly useful when:

1. A visualization isn't being generated correctly
2. The wrong visualization type is being detected
3. Parameters aren't being extracted correctly
4. You're developing new visualization detection patterns

### Example Usage

Using curl:

```bash
curl -X POST http://localhost:8000/api/nlp-visualization/debug \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a 3D surface plot of z = sin(x)*cos(y)"}'
```

Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/nlp-visualization/debug",
    json={"prompt": "Create a 3D surface plot of z = sin(x)*cos(y)"},
    headers={"Content-Type": "application/json"}
)

debug_info = response.json()
print(f"Detection method: {debug_info['detection_method']}")
print(f"Visualization type: {debug_info['visualization_type']}")

# Check pattern matching logs
for log in debug_info.get('pattern_match_logs', []):
    print(f"{log['level']}: {log['message']}")
```

## Common Issues and Troubleshooting

### 1. Wrong Visualization Type

If your prompt is detected as the wrong visualization type:

- Check if your prompt has clear indicators for the expected type
- Look at the pattern matching logs to see what patterns were matched
- Consider adding more specific keywords to your prompt
- Look for ambiguous phrases that might trigger another visualization type

### 2. No Visualization Type Detected

If no visualization type is detected:

- Check if you're using terminology the system recognizes
- Look at both pattern matching and LLM logs for clues
- Try rewording your prompt with more common terminology
- Verify you're requesting a supported visualization type

### 3. Parameters Not Extracted Correctly

If parameters (like expressions or ranges) aren't extracted correctly:

- Check the logs for extraction attempts
- Use more standard notation (e.g., use "x^2" instead of "x squared")
- Provide clearer parameter specifications (e.g., "from -5 to 5")
- Verify the expression syntax matches Python notation

## Advanced Debugging

For more advanced debugging, the endpoint also returns internal logs that show:

- Which patterns were attempted
- What expressions were extracted
- How parameters were processed
- Why certain detection paths were chosen over others

These logs can be invaluable when developing new visualization types or refining existing pattern matching logic.

## Using the Test Scripts

The repository includes test scripts to systematically test visualization detection:

- `debug_visualization_detection.py` - Tests a comprehensive suite of prompts
- `test_improved_viz_detection.py` - Tests specifically improved detection logic
- `test_direct_nlp_viz.py` - Tests direct API calls to the visualization endpoint

Run these scripts to verify detection performance across different visualization types. 
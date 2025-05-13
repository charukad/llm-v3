# LMStudio Integration for Mathematical Multimodal LLM System

This document explains how to use LMStudio as the inference backend for the Mathematical Multimodal LLM System. LMStudio provides a user-friendly interface for running local LLMs with optimized inference, making it easier to work with mathematical language models.

## Prerequisites

1. Install [LMStudio](https://lmstudio.ai/) - A user-friendly desktop application for running local LLMs
2. Ensure you have Python 3.8+ installed
3. Install all required dependencies for the Mathematical Multimodal LLM System

## Setup

1. **Start LMStudio and load the model:**
   - Launch LMStudio
   - Download/add the Mistral 7B Instruct model (or another compatible model)
   - Start the local server in LMStudio by clicking "Local Server" in the sidebar
   - Ensure the API server is running (typically at http://127.0.0.1:1234)

2. **Configure the Mathematical LLM System:**
   - The system will automatically use LMStudio if available
   - You can explicitly configure it with environment variables or command line arguments

## Usage Options

You have several ways to interact with the system:

### 1. Simple Chat Interface

Use the chat application to interact with the model for mathematical queries:

```bash
python chat_with_lmstudio.py --mode math
```

Options:
- `--mode [chat|math|algebra|calculus|statistics|linear_algebra|geometry]`: Select the domain expertise
- `--chain-of-thought`: Enable step-by-step reasoning
- `--model NAME`: Specify the model name in LMStudio
- `--url URL`: LMStudio API URL (default: http://127.0.0.1:1234)
- `--temperature FLOAT`: Control randomness (lower is more deterministic)
- `--max-tokens INT`: Maximum response length
- `--stream`: Stream tokens as they're generated
- `--save`: Save the conversation to a file

### 2. Testing LMStudio Integration

Test the LMStudio integration components:

```bash
python test_lmstudio.py
```

Options:
- `--url URL`: LMStudio API URL
- `--model NAME`: Model name in LMStudio
- `--tests [all|direct|engine|agent]`: Which test components to run

### 3. Running the Full System with LMStudio

Launch the complete Mathematical Multimodal LLM System with LMStudio:

```bash
python run_with_lmstudio.py
```

Options:
- `--lmstudio-url URL`: LMStudio API URL
- `--lmstudio-model NAME`: Model name in LMStudio
- `--host HOST`: Server host (default: 0.0.0.0)
- `--port PORT`: Server port (default: 8000)
- `--debug`: Enable debug mode

## Available Models in LMStudio

The system works best with these models:

1. `mistral-7b-instruct-v0.3` - Good balance of performance and speed
2. `llama-2-13b-chat` - Larger model, more capabilities but slower
3. `mixtral-8x7b-instruct-v0.1` - Advanced MoE model with excellent performance

## Using the API

When using LMStudio, the Mathematical Multimodal LLM System API is available at:

```
http://HOST:PORT/api/v1/
```

Example endpoints:
- `/api/v1/math/solve` - Solve mathematical problems
- `/api/v1/math/explain` - Get step-by-step explanations
- `/api/v1/multimodal/process` - Process math in images

## Troubleshooting

1. **LMStudio not found**
   - Ensure LMStudio is running and the API server is enabled
   - Check the URL (default: http://127.0.0.1:1234)
   - Verify the model is loaded in LMStudio

2. **Model not generating proper math responses**
   - Try a different model more suited for mathematical reasoning
   - Use lower temperature settings (0.1-0.3) for more deterministic responses
   - Enable chain-of-thought prompting for complex problems

3. **System errors**
   - Check the logs in the `logs/` directory
   - Ensure all dependencies are installed
   - Verify the MongoDB connection if using the full system

## Performance Optimization

- Use CUDA/GPU for faster inference if available in LMStudio
- Consider 4-bit quantization for larger models to reduce memory usage
- For complex mathematical visualization, increase timeout settings

## Examples

### Simple Math Query
```
python chat_with_lmstudio.py --mode math
You: Solve the equation 3x^2 - 12 = 0
```

### Domain-Specific Query
```
python chat_with_lmstudio.py --mode calculus
You: Find the derivative of f(x) = sin(x^2) * log(x)
```

### Using the Full System API
```python
import requests

response = requests.post("http://localhost:8000/api/v1/math/solve", 
                        json={"query": "Solve for x: 3x^2 - 12 = 0"})
result = response.json()
print(result["solution"])
``` 
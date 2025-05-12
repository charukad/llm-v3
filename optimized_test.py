#!/usr/bin/env python3
"""
Optimized script for testing the Mistral GGUF model with better performance.
"""

from llama_cpp import Llama
import time

# Define the prompt
prompt = "Calculate the derivative of f(x) = 3x^2 + 2x - 5"

# Start timing
start_time = time.time()

print("Loading model... This may take a minute.")

# Initialize the model with optimized settings
llm = Llama(
    model_path="models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=512,           # Smaller context window for faster loading
    n_threads=2,         # Fewer threads can sometimes be faster
    n_batch=512,         # Increase batch size
    use_mlock=True,      # Lock memory to prevent swapping
    use_mmap=True,       # Use memory mapping
    verbose=False        # Reduce verbosity
)

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

# Generate a response
print(f"Sending prompt: {prompt}")
print("Generating response...")

# Generate without streaming for faster results
generation_start = time.time()

response = llm(
    f"You are a helpful math assistant. Please solve: {prompt}",
    max_tokens=100,       # Lower token count for faster generation
    temperature=0.1,
    top_p=0.9,
    repeat_penalty=1.1,
    echo=False
)

generation_time = time.time() - generation_start

# Output results
print("\n--- MODEL RESPONSE ---\n")
print(response["choices"][0]["text"])
print("\n---------------------\n")
print(f"Response generated in {generation_time:.2f} seconds")
print(f"Total time: {time.time() - start_time:.2f} seconds") 
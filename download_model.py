#!/usr/bin/env python3
"""
Script to download and quantize an open-source LLM model for the Mathematical Multimodal LLM System.
"""

import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    parser = argparse.ArgumentParser(description="Download and quantize LLM model")
    parser.add_argument("--model_id", type=str, default="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", 
                        help="Model ID on Hugging Face")
    parser.add_argument("--output_dir", type=str, default="models/mistral-7b-v0.1-4bit", 
                        help="Output directory for the model")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (4 or 8)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading model {args.model_id}...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(args.output_dir)
    print("Tokenizer saved successfully!")
    
    # Download and quantize model
    if args.bits == 4:
        print("Downloading and quantizing model to 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    elif args.bits == 8:
        print("Downloading and quantizing model to 8-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        print("Downloading model in full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    
    # Save model
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Create a config file that instructs to load the model with 4-bit quantization
    with open(os.path.join(args.output_dir, "config_4bit.json"), "w") as f:
        f.write('{"load_in_4bit": true, "torch_dtype": "float16"}')
    
    print("Download and quantization completed successfully!")

if __name__ == "__main__":
    main() 
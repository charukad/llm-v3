"""
Inference module for Mistral 7B.

This module provides optimized inference using vLLM for the Mistral 7B model.
It also supports API-based inference through LMStudio.
"""
import os
import logging
import time
import json
import requests
from typing import List, Dict, Any, Optional, Union
import torch
from threading import Thread

logger = logging.getLogger(__name__)

# Try to import vLLM for optimized inference (only used when not using LMStudio)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    # This is expected and normal when using LMStudio
    VLLM_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class LMStudioInference:
    """
    Inference using LMStudio API.
    
    This class provides generation capabilities through the LMStudio API.
    """
    
    def __init__(
        self, 
        api_url: str = "http://127.0.0.1:1234",
        model_name: str = "mistral-7b-instruct-v0.3",
        max_tokens: int = 2048
    ):
        """
        Initialize the LMStudio inference client.
        
        Args:
            api_url: URL of the LMStudio API
            model_name: Name of the model in LMStudio
            max_tokens: Maximum number of tokens for generation
        """
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.complete_url = f"{self.api_url}/v1/completions"
        
        # Verify API connection
        try:
            response = requests.get(f"{self.api_url}/v1/models")
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Connected to LMStudio API. Available models: {models}")
            else:
                logger.warning(f"LMStudio API responded with status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to LMStudio API at {self.api_url}: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, List[str]]:
        """
        Generate text using the LMStudio API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            stop_sequences: Sequences that stop generation
            stream: Whether to stream the output
            
        Returns:
            Generated text, or a list of streamed outputs if stream=True
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "stream": stream
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        # Log request
        logger.info(f"Sending request to LMStudio API with {len(prompt)} chars")
        start_time = time.time()
        
        try:
            if stream:
                return self._generate_stream(payload)
            else:
                return self._generate_complete(payload)
        except Exception as e:
            logger.error(f"Error in LMStudio API request: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_complete(self, payload: Dict[str, Any]) -> str:
        """
        Complete a prompt with non-streaming response.
        
        Args:
            payload: Request payload
            
        Returns:
            Generated text
        """
        response = requests.post(self.complete_url, json=payload)
        generation_time = time.time() - payload.get("start_time", time.time())
        
        if response.status_code != 200:
            logger.error(f"LMStudio API error: {response.status_code} {response.text}")
            return f"API Error: {response.status_code}"
        
        result = response.json()
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        # Extract the response text
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["text"]
        else:
            logger.warning("Unexpected response format from LMStudio API")
            return ""
    
    def _generate_stream(self, payload: Dict[str, Any]) -> List[str]:
        """
        Complete a prompt with streaming response.
        
        Args:
            payload: Request payload
            
        Returns:
            List of text chunks from the stream
        """
        response = requests.post(self.complete_url, json=payload, stream=True)
        
        if response.status_code != 200:
            logger.error(f"LMStudio API streaming error: {response.status_code} {response.text}")
            return [f"API Error: {response.status_code}"]
        
        chunks = []
        start_time = time.time()
        
        for line in response.iter_lines():
            if line:
                try:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        line_json = json.loads(line_text[6:])
                        if "choices" in line_json and len(line_json["choices"]) > 0:
                            chunk = line_json["choices"][0]["text"]
                            chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Error parsing streaming response: {e}")
        
        generation_time = time.time() - start_time
        logger.info(f"Streaming generation completed in {generation_time:.2f}s")
        
        return chunks

class InferenceEngine:
    """
    Inference engine for optimized language model inference.
    
    Provides multiple implementation options:
    1. LMStudio API
    2. vLLM (optimized inference)
    3. Transformers (standard inference)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_tokens: int = 2048,
        quantization: Optional[str] = None,
        use_vllm: bool = False,
        use_lmstudio: bool = True,
        lmstudio_url: str = "http://127.0.0.1:1234",
        lmstudio_model: str = "mistral-7b-instruct-v0.3",
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model or model identifier
            device: Device to run inference on ("cuda", "cpu", or "auto")
            max_tokens: Maximum number of tokens for generation
            quantization: Quantization method if using transformers
            use_vllm: Whether to use vLLM if available
            use_lmstudio: Whether to use LMStudio API
            lmstudio_url: URL of the LMStudio API
            lmstudio_model: Name of the model in LMStudio
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set up inference based on selected method
        if use_lmstudio:
            logger.info(f"Using LMStudio API at {lmstudio_url} with model {lmstudio_model}")
            self.inference_type = "lmstudio"
            self.lmstudio = LMStudioInference(
                api_url=lmstudio_url,
                model_name=lmstudio_model,
                max_tokens=max_tokens
            )
        elif use_vllm and VLLM_AVAILABLE and self.device == "cuda":
            logger.info("Using vLLM for inference")
            self.inference_type = "vllm"
            self._setup_vllm()
        else:
            if use_vllm and not VLLM_AVAILABLE:
                # No message needed since we're primarily using LMStudio
                pass
            elif use_vllm and self.device != "cuda":
                # No message needed since we're primarily using LMStudio
                pass
            
            logger.info("Using transformers for inference")
            self.inference_type = "transformers"
            self._setup_transformers(quantization)
    
    def _setup_vllm(self):
        """Set up vLLM for inference."""
        gpu_memory_utilization = 0.9  # Use 90% of GPU memory by default
        
        try:
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
            logger.info("vLLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.warning("Falling back to transformers")
            self.inference_type = "transformers"
            self._setup_transformers()
    
    def _setup_transformers(self, quantization: Optional[str] = None):
        """
        Set up transformers for inference.
        
        Args:
            quantization: Quantization method to use
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Setting up transformers for inference")
        
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        # Handle quantization
        if quantization == "4bit" and self.device == "cuda":
            load_kwargs.update({
                "load_in_4bit": True,
                "device_map": "auto",
                "quantization_config": {
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                },
            })
        elif quantization == "8bit" and self.device == "cuda":
            load_kwargs.update({
                "load_in_8bit": True,
                "device_map": "auto",
            })
        else:
            load_kwargs.update({
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            })
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        
        if self.device == "cuda":
            logger.info("Using CUDA for inference")
        else:
            logger.info("Using CPU for inference. This will be slow for large models.")
            self.model = self.model.to(self.device)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Transformers initialized successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, List[str]]:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides instance default)
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            stop_sequences: Sequences that stop generation
            stream: Whether to stream the output
            
        Returns:
            Generated text, or a list of streamed outputs if stream=True
        """
        if self.inference_type == "lmstudio":
            return self.lmstudio.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
                stream=stream,
            )
        elif self.inference_type == "vllm":
            return self._generate_vllm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop_sequences,
                stream=stream,
            )
        else:
            return self._generate_transformers(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
                stream=stream,
            )
    
    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, List[str]]:
        """
        Generate text using vLLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            stop: Sequences that stop generation
            stream: Whether to stream the output
            
        Returns:
            Generated text, or a list of streamed outputs if stream=True
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        # Generate
        start_time = time.time()
        
        if stream:
            streamed_outputs = []
            for output in self.llm.generate(prompt, sampling_params, stream=True):
                # For simplicity, we're just collecting the outputs
                # In a real app, you'd yield these to the client
                streamed_outputs.append(output.outputs[0].text)
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f}s")
            return streamed_outputs
        else:
            outputs = self.llm.generate(prompt, sampling_params)
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f}s")
            
            # Extract the generated text
            return outputs[0].outputs[0].text
    
    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, List[str]]:
        """
        Generate text using transformers.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            stop_sequences: Sequences that stop generation
            stream: Whether to stream the output
            
        Returns:
            Generated text, or a list of streamed outputs if stream=True
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device != "cpu":
            input_ids = input_ids.to(self.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Handle stop sequences
        if stop_sequences:
            stop_token_ids = [self.tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
            # Flatten the list
            stop_token_ids = [id for sublist in stop_token_ids for id in sublist]
            gen_kwargs["eos_token_id"] = stop_token_ids
        
        start_time = time.time()
        
        if stream:
            streamed_outputs = []
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            gen_kwargs["streamer"] = streamer
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs={
                "input_ids": input_ids,
                **gen_kwargs,
            })
            thread.start()
            
            # Collect streamed outputs
            for text in streamer:
                streamed_outputs.append(text)
            
            thread.join()
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f}s")
            return streamed_outputs
        else:
            output_ids = self.model.generate(input_ids, **gen_kwargs)
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f}s")
            
            # Decode the output, skipping the prompt
            output_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Check for stop sequences if manually specified
            if stop_sequences:
                for seq in stop_sequences:
                    if seq in output_text:
                        output_text = output_text[:output_text.find(seq)]
            
            return output_text

#!/usr/bin/env python3
"""
Interactive chat application using LMStudio integration for the Mathematical LLM System.

This script provides a command-line interface for chatting with the LLM
through the LMStudio API, with special support for mathematical queries.
"""
import logging
import os
import sys
import time
import argparse
import readline  # For better command-line input handling
from typing import List, Tuple, Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("lmstudio_chat.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from core.mistral.inference import InferenceEngine, LMStudioInference
from core.prompting.system_prompts import get_system_prompt
from core.prompting.chain_of_thought import get_cot_template

def setup_argparse() -> argparse.Namespace:
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Chat with the Mathematical LLM System using LMStudio API")
    parser.add_argument("--url", default="http://127.0.0.1:1234",
                      help="URL of the LMStudio API server")
    parser.add_argument("--model", default="mistral-7b-instruct-v0.3",
                      help="Model name in LMStudio")
    parser.add_argument("--max-tokens", type=int, default=1000,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                      help="Sampling temperature (lower for math problems)")
    parser.add_argument("--mode", choices=["chat", "math", "algebra", "calculus", 
                                          "statistics", "linear_algebra", "geometry"],
                      default="math", help="Chat mode or specific math domain")
    parser.add_argument("--chain-of-thought", action="store_true",
                      help="Enable chain-of-thought prompting for mathematical problems")
    parser.add_argument("--stream", action="store_true",
                      help="Stream the response token by token")
    parser.add_argument("--save", action="store_true",
                      help="Save conversation history to a file")
    return parser.parse_args()

def create_math_prompt(user_input: str, history: List[Tuple[str, str]], 
                     domain: str = "general", use_cot: bool = False) -> str:
    """
    Create a prompt for the math assistant mode.
    
    Args:
        user_input: The user's query
        history: Conversation history as (question, answer) pairs
        domain: Mathematical domain
        use_cot: Whether to use chain-of-thought prompting
        
    Returns:
        Formatted prompt
    """
    # Get appropriate system prompt and CoT template
    system_prompt = get_system_prompt(domain)
    cot_template = get_cot_template(domain) if use_cot else ""
    
    # Format conversation history
    context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])
    if context:
        context += "\n"
    
    # Create the prompt with optional CoT
    if use_cot:
        prompt = f"{system_prompt}\n\n{context}User: {user_input}\nAssistant: {cot_template}"
    else:
        prompt = f"{system_prompt}\n\n{context}User: {user_input}\nAssistant:"
    
    return prompt

def create_chat_prompt(user_input: str, history: List[Tuple[str, str]]) -> str:
    """
    Create a prompt for the general chat mode.
    
    Args:
        user_input: The user's query
        history: Conversation history as (question, answer) pairs
        
    Returns:
        Formatted prompt
    """
    system_prompt = """You are a helpful, intelligent assistant with expertise in mathematics, science, and general knowledge. 
You provide clear, accurate, and thoughtful responses to questions."""
    
    # Format conversation history
    context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])
    if context:
        context += "\n"
    
    prompt = f"{system_prompt}\n\n{context}User: {user_input}\nAssistant:"
    return prompt

def save_conversation(history: List[Tuple[str, str]], mode: str) -> None:
    """
    Save the conversation history to a file.
    
    Args:
        history: Conversation history as (question, answer) pairs
        mode: Chat mode used
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{mode}_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(f"=== LMStudio Chat Conversation ({mode.upper()} mode) ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, (question, answer) in enumerate(history, 1):
            f.write(f"[{i}] User: {question}\n\n")
            f.write(f"[{i}] Assistant: {answer}\n\n")
            f.write("-" * 50 + "\n\n")
    
    logger.info(f"Conversation saved to {filename}")
    return filename

def main() -> None:
    """Run the chat application."""
    args = setup_argparse()
    
    logger.info(f"Starting chat with LMStudio at {args.url} with model {args.model}")
    logger.info(f"Mode: {args.mode}")
    
    # Create the inference engine
    inference = InferenceEngine(
        model_path="placeholder",  # Not used with LMStudio
        use_lmstudio=True,
        lmstudio_url=args.url,
        lmstudio_model=args.model
    )
    
    history = []
    
    # Display welcome header
    print("\n\033[1;34m=== Mathematical LLM System Chat Interface ===\033[0m")
    print(f"Mode: \033[1;36m{args.mode.upper()}\033[0m")
    print(f"Model: \033[1;36m{args.model}\033[0m")
    print(f"Chain-of-Thought: \033[1;36m{'Enabled' if args.chain_of_thought else 'Disabled'}\033[0m")
    print(f"Temperature: \033[1;36m{args.temperature}\033[0m")
    print()
    print("\033[1mCommands:\033[0m")
    print("  \033[33mexit, quit\033[0m - Exit the application")
    print("  \033[33mclear\033[0m - Clear conversation history")
    print("  \033[33msave\033[0m - Save the conversation to a file")
    print("  \033[33mhelp\033[0m - Show these commands")
    print("  \033[33mmode <mode>\033[0m - Switch mode (chat, math, algebra, calculus, etc.)")
    print("-" * 50)
    
    while True:
        try:
            # Get user input with a colorful prompt
            user_input = input("\033[1;32mYou:\033[0m ").strip()
            
            # Check for commands
            if user_input.lower() in ["exit", "quit"]:
                # Save before exiting if requested
                if args.save and history:
                    save_conversation(history, args.mode)
                print("Goodbye!")
                break
            
            # Clear command
            elif user_input.lower() == "clear":
                history = []
                print("Conversation history cleared.")
                continue
            
            # Save command
            elif user_input.lower() == "save":
                if history:
                    filename = save_conversation(history, args.mode)
                    print(f"Conversation saved to \033[1m{filename}\033[0m")
                else:
                    print("No conversation to save.")
                continue
            
            # Help command
            elif user_input.lower() == "help":
                print("\033[1mAvailable commands:\033[0m")
                print("  \033[33mexit, quit\033[0m - Exit the application")
                print("  \033[33mclear\033[0m - Clear conversation history")
                print("  \033[33msave\033[0m - Save the conversation to a file")
                print("  \033[33mhelp\033[0m - Show these commands")
                print("  \033[33mmode <mode>\033[0m - Switch mode (chat, math, algebra, calculus, etc.)")
                continue
            
            # Mode switch command
            elif user_input.lower().startswith("mode "):
                new_mode = user_input.lower().split("mode ")[1].strip()
                valid_modes = ["chat", "math", "algebra", "calculus", "statistics", "linear_algebra", "geometry"]
                
                if new_mode in valid_modes:
                    args.mode = new_mode
                    print(f"Switched to \033[1;36m{args.mode.upper()}\033[0m mode.")
                else:
                    print(f"Invalid mode. Choose from: {', '.join(valid_modes)}")
                continue
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Create prompt based on mode
            if args.mode != "chat":
                prompt = create_math_prompt(
                    user_input, 
                    history, 
                    domain=args.mode, 
                    use_cot=args.chain_of_thought
                )
            else:
                prompt = create_chat_prompt(user_input, history)
            
            # Generate response
            start_time = time.time()
            print("\033[1;33mAssistant:\033[0m", end=" " if not args.stream else "\n")
            sys.stdout.flush()
            
            if args.stream:
                # Stream the response
                response_chunks = inference.generate(
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=True
                )
                
                full_response = ""
                for chunk in response_chunks:
                    print(chunk, end="")
                    sys.stdout.flush()
                    full_response += chunk
                
                print()  # Add a newline at the end
                response = full_response
            else:
                # Generate the complete response
                response = inference.generate(
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                print(response)
            
            end_time = time.time()
            print(f"\033[90m(Generated in {end_time - start_time:.2f} seconds)\033[0m")
            
            # Add to history
            history.append((user_input, response))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            # Save before exiting if requested
            if args.save and history:
                save_conversation(history, args.mode)
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print(f"\033[91mError: {e}\033[0m")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        print(f"\033[91mUnhandled error: {e}\033[0m") 
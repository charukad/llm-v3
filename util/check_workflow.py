#!/usr/bin/env python
"""
Utility script to check the status and results of a workflow.

Usage:
    python check_workflow.py <workflow_id> [--watch] [--interval SECONDS]
"""
import sys
import asyncio
import json
import argparse
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import textwrap

# Add parent directory to path to allow imports
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestration.manager.orchestration_manager import get_orchestration_manager
from orchestration.workflow.workflow_registry import WorkflowStatus
from multimodal.context.context_manager import get_context_manager

# Terminal colors for better visualization
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
}

# Progress tracking
last_seen_steps = 0
last_step_status = {}

async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get the status and results of a workflow.
    
    Args:
        workflow_id: The ID of the workflow to check
        
    Returns:
        Dictionary containing workflow status and data
    """
    # Get the orchestration manager
    orchestration_mgr = get_orchestration_manager()
    
    # Get the workflow context
    workflow = await orchestration_mgr.get_workflow(workflow_id)
    
    if not workflow:
        return {
            "success": False,
            "error": f"Workflow {workflow_id} not found",
        }
    
    # Get conversation context if available
    context_data = None
    if workflow.conversation_id:
        try:
            context_manager = get_context_manager()
            context_data = await context_manager.get_conversation_context(
                workflow.conversation_id, 
                workflow.data.get("context_id")
            )
        except Exception as e:
            print(f"Failed to retrieve conversation context: {e}")
    
    # Return the workflow context data
    return {
        "success": True,
        "workflow_id": workflow_id,
        "status": workflow.status,
        "data": workflow.data,
        "steps": workflow.steps,
        "completed_at": workflow.completed_at,
        "conversation_id": workflow.conversation_id,
        "context_data": context_data,
        "created_at": workflow.created_at,
        "updated_at": workflow.updated_at,
    }

def format_step_progression(steps: List[Dict[str, Any]]) -> str:
    """
    Format the step progression information with visual indicators.
    
    Args:
        steps: List of workflow steps
        
    Returns:
        Formatted string showing step progression
    """
    global last_seen_steps, last_step_status
    
    lines = []
    current_steps = len(steps)
    new_steps = current_steps > last_seen_steps
    last_seen_steps = current_steps
    
    for i, step in enumerate(steps):
        name = step.get('name', 'Unknown')
        description = step.get('description', 'No description')
        agent = step.get('agent', 'Unknown')
        step_id = f"{i+1}.{name}"
        
        # Check step status
        is_completed = 'completed_at' in step
        is_new = step_id not in last_step_status
        just_completed = step_id in last_step_status and not last_step_status[step_id] and is_completed
        
        # Update status tracking
        last_step_status[step_id] = is_completed
        
        # Format the step with appropriate indicators
        status_icon = "✓" if is_completed else "⋯"
        color_start = COLORS["GREEN"] if is_completed else COLORS["YELLOW"]
        
        # Add special indicators for new or just completed steps
        if is_new and new_steps:
            status_icon = "+"
            color_start = COLORS["CYAN"]
        elif just_completed:
            status_icon = "✓"
            color_start = COLORS["GREEN"] + COLORS["BOLD"]
            
        # Calculate elapsed time for completed steps
        elapsed = ""
        if is_completed and 'started_at' in step and 'completed_at' in step:
            try:
                start_time = datetime.fromisoformat(step['started_at'])
                end_time = datetime.fromisoformat(step['completed_at'])
                duration = (end_time - start_time).total_seconds()
                elapsed = f" ({duration:.2f}s)"
            except (ValueError, TypeError):
                pass
                
        # Format the line
        step_line = f"  {color_start}{status_icon} [{i+1}] {name}{COLORS['RESET']} ({agent}): {description}{elapsed}"
        lines.append(step_line)
        
        # Add result information for completed steps
        if is_completed and 'result' in step:
            result_snippet = json.dumps(step['result'])[:100]
            if len(result_snippet) == 100:
                result_snippet += "..."
            lines.append(f"    → Result: {result_snippet}")
    
    return "\n".join(lines)

def format_result_details(result: Dict[str, Any], agent_type: str) -> str:
    """
    Format the result details based on agent type.
    
    Args:
        result: Result data
        agent_type: Type of agent that generated the result
        
    Returns:
        Formatted string showing result details
    """
    if not result:
        return "No results available yet."
        
    lines = []
    
    # Format based on agent type
    if agent_type == "math_computation":
        # Math computation results
        if "latex_result" in result:
            lines.append(f"{COLORS['BOLD']}Mathematical Result:{COLORS['RESET']}")
            lines.append(f"  {result['latex_result']}")
        
        if "steps" in result:
            lines.append(f"\n{COLORS['BOLD']}Solution Steps:{COLORS['RESET']}")
            for i, step in enumerate(result["steps"]):
                lines.append(f"  {COLORS['CYAN']}Step {i+1}{COLORS['RESET']}: {step.get('description', '')}")
                if "latex" in step:
                    lines.append(f"    {step['latex']}")
                    
    elif agent_type == "visualization":
        # Visualization results
        lines.append(f"{COLORS['BOLD']}Visualization:{COLORS['RESET']}")
        
        if "visualization_url" in result:
            lines.append(f"  URL: {result['visualization_url']}")
            
        if "image_data" in result:
            lines.append(f"  Image data available ({len(result['image_data'])} bytes)")
            
        if "code" in result:
            lines.append(f"\n{COLORS['BOLD']}Visualization Code:{COLORS['RESET']}")
            code_lines = result["code"].split('\n')
            formatted_code = '\n'.join([f"  {line}" for line in code_lines[:15]])
            if len(code_lines) > 15:
                formatted_code += f"\n  ... ({len(code_lines) - 15} more lines)"
            lines.append(formatted_code)
            
    elif agent_type == "core_llm":
        # LLM results
        if "response" in result:
            lines.append(f"{COLORS['BOLD']}LLM Response:{COLORS['RESET']}")
            wrapped_response = textwrap.fill(result["response"], width=100)
            formatted_response = '\n'.join([f"  {line}" for line in wrapped_response.split('\n')])
            lines.append(formatted_response)
    
    else:
        # Generic result display
        lines.append(f"{COLORS['BOLD']}Agent Result:{COLORS['RESET']}")
        
        # Display key fields with special formatting
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            elif isinstance(value, dict) or isinstance(value, list):
                value = str(type(value)) + f" with {len(value)} items"
            lines.append(f"  {COLORS['CYAN']}{key}{COLORS['RESET']}: {value}")
    
    return "\n".join(lines)

def format_conversation_context(context_data: Optional[Dict[str, Any]]) -> str:
    """
    Format the conversation context information.
    
    Args:
        context_data: Context data from the context manager
        
    Returns:
        Formatted string showing context information
    """
    if not context_data:
        return "No context data available."
        
    lines = []
    lines.append(f"{COLORS['BOLD']}Conversation Context:{COLORS['RESET']}")
    lines.append(f"  Context ID: {context_data.get('context_id', 'Unknown')}")
    
    # Show conversation history
    conversation_history = context_data.get("conversation_history", [])
    if conversation_history:
        lines.append(f"\n{COLORS['BOLD']}Conversation History ({len(conversation_history)} turns):{COLORS['RESET']}")
        # Show last 3 turns at most
        for i, turn in enumerate(conversation_history[-3:]):
            if "user" in turn:
                user_msg = turn["user"]
                if len(user_msg) > 100:
                    user_msg = user_msg[:97] + "..."
                lines.append(f"  {COLORS['YELLOW']}User{COLORS['RESET']}: {user_msg}")
            if "assistant" in turn:
                assistant_msg = turn["assistant"]
                if len(assistant_msg) > 100:
                    assistant_msg = assistant_msg[:97] + "..."
                lines.append(f"  {COLORS['GREEN']}Assistant{COLORS['RESET']}: {assistant_msg}")
    else:
        lines.append("  No conversation history available.")
    
    return "\n".join(lines)

def display_workflow_results(workflow_data: Dict[str, Any], show_context: bool = False) -> None:
    """
    Display the workflow results in a user-friendly format in the terminal.
    
    Args:
        workflow_data: The workflow data to display
        show_context: Whether to show conversation context
    """
    if not workflow_data.get("success", False):
        print(f"{COLORS['RED']}Error: {workflow_data.get('error', 'Unknown error')}{COLORS['RESET']}")
        return
    
    # Display basic workflow info
    print(f"\n{COLORS['BOLD']}{'=' * 80}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}Workflow ID:{COLORS['RESET']} {workflow_data['workflow_id']}")
    
    # Status with color
    status = workflow_data['status']
    status_color = COLORS["YELLOW"]
    if status == WorkflowStatus.COMPLETED:
        status_color = COLORS["GREEN"]
    elif status == WorkflowStatus.FAILED:
        status_color = COLORS["RED"]
    
    print(f"{COLORS['BOLD']}Status:{COLORS['RESET']} {status_color}{status}{COLORS['RESET']}")
    
    # Time information
    if workflow_data.get('created_at'):
        created_time = datetime.fromisoformat(workflow_data['created_at'])
        print(f"{COLORS['BOLD']}Created:{COLORS['RESET']} {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if workflow_data.get('completed_at'):
        completed_time = datetime.fromisoformat(workflow_data['completed_at'])
        print(f"{COLORS['BOLD']}Completed:{COLORS['RESET']} {completed_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate duration if both times are available
        if workflow_data.get('created_at'):
            created_time = datetime.fromisoformat(workflow_data['created_at'])
            duration = (completed_time - created_time).total_seconds()
            print(f"{COLORS['BOLD']}Duration:{COLORS['RESET']} {duration:.2f} seconds")
    
    # Display conversation ID
    if workflow_data.get('conversation_id'):
        print(f"{COLORS['BOLD']}Conversation ID:{COLORS['RESET']} {workflow_data['conversation_id']}")
    
    # Display agent and input type
    data = workflow_data.get('data', {})
    agent_type = data.get('routing', {}).get('agent_type', 'unknown')
    input_type = data.get('input_type', 'unknown')
    
    print(f"{COLORS['BOLD']}Agent Type:{COLORS['RESET']} {agent_type}")
    print(f"{COLORS['BOLD']}Input Type:{COLORS['RESET']} {input_type}")
    
    print(f"{COLORS['BOLD']}{'-' * 80}{COLORS['RESET']}")
    
    # Content preview
    content = data.get('content', '')
    if content:
        if isinstance(content, str):
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"{COLORS['BOLD']}Content:{COLORS['RESET']} {content}")
        else:
            print(f"{COLORS['BOLD']}Content:{COLORS['RESET']} [Complex content of type {type(content).__name__}]")
    
    # Display steps information with progression
    steps = workflow_data.get('steps', [])
    if steps:
        print(f"\n{COLORS['BOLD']}Steps Progress ({len(steps)}):{COLORS['RESET']}")
        print(format_step_progression(steps))
    
    # Display the results
    result = data.get('result', {})
    
    print(f"\n{COLORS['BOLD']}{'-' * 80}{COLORS['RESET']}")
    
    # Show agent instructions
    instructions = data.get('instructions', {})
    if instructions:
        print(f"{COLORS['BOLD']}Agent Instructions:{COLORS['RESET']}")
        
        # Display title and description
        if 'title' in instructions:
            print(f"  {COLORS['CYAN']}Title:{COLORS['RESET']} {instructions['title']}")
        if 'description' in instructions:
            print(f"  {COLORS['CYAN']}Description:{COLORS['RESET']} {instructions['description']}")
        
        # Display step-by-step instructions in condensed form
        if 'step_by_step' in instructions and isinstance(instructions['step_by_step'], list):
            step_count = len(instructions['step_by_step'])
            print(f"  {COLORS['CYAN']}Steps:{COLORS['RESET']} {step_count} step-by-step instructions provided")
        
        print()
    
    # Check workflow status and display appropriate results
    if status == WorkflowStatus.COMPLETED:
        print(f"{COLORS['GREEN']}{COLORS['BOLD']}Workflow completed successfully!{COLORS['RESET']}")
        print(f"\n{COLORS['BOLD']}Results:{COLORS['RESET']}")
        print(format_result_details(result, agent_type))
        
        # Show explanation if available
        explanation = data.get('explanation')
        if explanation:
            print(f"\n{COLORS['BOLD']}LLM Explanation:{COLORS['RESET']}")
            wrapped_explanation = textwrap.fill(explanation, width=100)
            formatted_explanation = '\n'.join([f"  {line}" for line in wrapped_explanation.split('\n')])
            print(formatted_explanation)
            
    elif status == WorkflowStatus.RUNNING:
        print(f"{COLORS['YELLOW']}Workflow is still running...{COLORS['RESET']}")
        
        # Show current agent processing progress
        current_step_idx = workflow_data.get("current_step_index", 0)
        if current_step_idx < len(steps):
            current_step = steps[current_step_idx]
            print(f"\n{COLORS['BOLD']}Current Step:{COLORS['RESET']} {current_step.get('name')}")
            
            # Show any partial results
            if result:
                print(f"\n{COLORS['BOLD']}Partial Results:{COLORS['RESET']}")
                print(format_result_details(result, agent_type))
                
    elif status == WorkflowStatus.FAILED:
        print(f"{COLORS['RED']}{COLORS['BOLD']}Workflow failed!{COLORS['RESET']}")
        
        # Show error if available
        error = workflow_data.get('error', {})
        if error:
            print(f"\n{COLORS['BOLD']}Error:{COLORS['RESET']} {error.get('message', 'Unknown error')}")
            if 'details' in error:
                print(f"{COLORS['BOLD']}Details:{COLORS['RESET']}")
                print(json.dumps(error.get('details', {}), indent=2))
    
    # Show context information if requested
    if show_context and workflow_data.get('context_data'):
        print(f"\n{COLORS['BOLD']}{'-' * 80}{COLORS['RESET']}")
        print(format_conversation_context(workflow_data.get('context_data')))
    
    # End display
    print(f"\n{COLORS['BOLD']}{'=' * 80}{COLORS['RESET']}")

async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check the status and results of a workflow.')
    parser.add_argument('workflow_id', type=str, help='The ID of the workflow to check')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch the workflow for updates')
    parser.add_argument('--interval', '-i', type=int, default=2, help='Refresh interval in seconds when watching (default: 2)')
    parser.add_argument('--context', '-c', action='store_true', help='Show conversation context information')
    
    args = parser.parse_args()
    
    if args.watch:
        # Watch mode - keep checking until workflow completes or fails
        try:
            while True:
                workflow_data = await get_workflow_status(args.workflow_id)
                
                # Clear terminal
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Show current time
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{COLORS['BLUE']}[{current_time}] Monitoring workflow: {args.workflow_id}{COLORS['RESET']}")
                
                # Display results
                display_workflow_results(workflow_data, args.context)
                
                # Exit if workflow is completed or failed
                if workflow_data.get('success', False) and workflow_data.get('status') in [
                    WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED
                ]:
                    break
                
                # Show waiting message
                print(f"\n{COLORS['YELLOW']}Refreshing in {args.interval} seconds... Press Ctrl+C to exit{COLORS['RESET']}")
                
                # Wait before checking again
                await asyncio.sleep(args.interval)
                
        except KeyboardInterrupt:
            print(f"\n{COLORS['YELLOW']}Monitoring stopped by user{COLORS['RESET']}")
            
    else:
        # One-time check
        workflow_data = await get_workflow_status(args.workflow_id)
        display_workflow_results(workflow_data, args.context)

if __name__ == "__main__":
    # Reset global tracking variables
    last_seen_steps = 0
    last_step_status = {}
    
    # Run main function
    asyncio.run(main()) 
"""
Central Input Agent for the multimodal system.

This agent serves as the central point of the entire multimodal system,
handling all incoming requests, determining the appropriate specialized agent,
and generating detailed instructions for processing each request.
"""
import logging
import json
import uuid
import requests
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from core.agent.llm_agent import CoreLLMAgent
from multimodal.unified_pipeline.content_router import ContentRouter
from multimodal.context.context_manager import get_context_manager
from orchestration.manager.orchestration_manager import get_orchestration_manager
from orchestration.agents.registry import get_agent_registry

logger = logging.getLogger(__name__)

class InputAgent:
    """
    Central Input Agent for the multimodal system.
    
    This agent serves as the primary entry point for all requests. It:
    1. Analyzes the input with an LLM to understand the user's intent
    2. Determines which specialized agent(s) should handle the request
    3. Generates detailed processing instructions for those agents
    4. Coordinates the execution of the request through the orchestration manager
    5. Maintains context across multiple interactions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Input Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize the core LLM agent for deep understanding of requests
        self.llm_agent = CoreLLMAgent(self.config.get("llm_config"))
        
        # Initialize content router for determining which agent should handle the request
        self.content_router = ContentRouter({"use_llm_router": True})
        
        # Get access to the context manager for maintaining conversation history
        self.context_manager = get_context_manager()
        
        # Get access to the agent registry for specialized agent capabilities
        self.agent_registry = get_agent_registry()
        
        # Define endpoint mapping for different agent types
        self.agent_endpoint_map = {
            "core_llm": "/llm/generate",
            "math_computation": "/math/compute",
            "visualization": "/visualization/generate",
            "ocr": "/ocr/process",
            "search": "/search/query",
            "text_processing": "/text/process"
        }
        
        # Define external visualization endpoint
        self.nlp_visualization_endpoint = "http://localhost:8000/nlp-visualization"
        
        # Define agent-specific instruction templates
        self.agent_instruction_templates = {
            "visualization": """
Generate a {plot_type} visualization with the following specifications:
- Title: "{title}"
- Data: {data_description}
- X-axis: {x_axis}
- Y-axis: {y_axis}
{additional_instructions}

Please follow these steps:
1. Prepare the data in the correct format
2. Set up the visualization environment
3. Create the base visualization
4. Add all required labels, legends, and annotations
5. Apply appropriate styling and colors
6. Return both the visualization and the code used to generate it
""",
            "math_computation": """
Solve the following mathematical problem:
{problem_statement}

Please follow these steps:
1. Parse the mathematical expression
2. Identify the appropriate solution method
3. Apply the method step-by-step
4. Provide intermediate results
5. Verify the solution
6. Return both the final answer and the solution steps in LaTeX format
""",
            "search": """
Perform a comprehensive search on:
{search_query}

Please follow these steps:
1. Break down the query into key concepts
2. Search across all available databases and sources
3. Prioritize results by relevance and credibility
4. Provide a summary of the top findings
5. Include direct citations and references
6. Return both a concise answer and supporting details
"""
        }
        
        logger.info("Initialized Central Input Agent")
    
    async def process_request(self, 
                             request_data: Dict[str, Any],
                             conversation_id: Optional[str] = None,
                             context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an incoming request.
        
        Args:
            request_data: The request data including input type and content
            conversation_id: Optional conversation ID for context tracking
            context_id: Optional context ID for specific context
            
        Returns:
            Processing result with workflow information
        """
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        logger.info(f"Processing request {request_id} [conversation: {conversation_id}]")
        
        try:
            # Step 1: Get conversation context if available
            context_data = None
            if conversation_id:
                try:
                    context_data = await self.context_manager.get_conversation_context(
                        conversation_id, context_id
                    )
                    logger.debug(f"Retrieved context for conversation {conversation_id}")
                except Exception as e:
                    logger.warning(f"Error retrieving context: {str(e)}")
            
            # Step 2: Extract input type and content
            input_type = request_data.get("input_type")
            content = request_data.get("content")
            
            if not input_type or not content:
                return {
                    "success": False,
                    "error": "Missing input_type or content in request",
                    "request_id": request_id
                }
            
            # Step 3: Route to appropriate agent using the content router
            processed_request = {
                "input_type": input_type,
                "text": content if input_type == "text" else None,
                "content": content,
            }
            
            routing_result = self.content_router.route_content(processed_request, context_data)
            
            # Step 4: Generate detailed instructions for the target agent
            target_agent_type = routing_result.get("agent_type", "core_llm")
            target_capabilities = routing_result.get("capabilities", [])
            
            # Step 5: Handle visualization requests by sending to NLP-Visualization endpoint
            if target_agent_type == "visualization" or self._is_visualization_request(content):
                try:
                    logger.info(f"Sending visualization request to NLP-Visualization endpoint: {content}")
                    viz_response = self._send_to_nlp_visualization(content)
                    
                    # Add visualization response to the workflow data
                    workflow_data = {
                        "request_id": request_id,
                        "input_type": input_type,
                        "content": content,
                        "routing": {
                            "agent_type": "visualization",
                            "capabilities": target_capabilities,
                            "confidence": routing_result.get("confidence", 1.0),
                            "reasoning": "Request routed to external visualization service"
                        },
                        "visualization_response": viz_response,
                        "conversation_id": conversation_id,
                        "context_id": context_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Calculate processing time
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    workflow_data["processing_time_ms"] = round(processing_time, 2)
                    
                    return {
                        "success": True,
                        "request_id": request_id,
                        "input_type": input_type,
                        "agent_type": "visualization",
                        "processing_time_ms": workflow_data["processing_time_ms"],
                        "status": "completed",
                        "visualization_data": viz_response
                    }
                    
                except Exception as e:
                    logger.error(f"Error sending to visualization endpoint: {str(e)}")
                    # Continue with normal processing if visualization endpoint fails
            
            # Step 6: Create detailed processing instructions with an LLM
            agent_instructions = await self._generate_agent_instructions(
                target_agent_type,
                processed_request,
                context_data
            )
            
            # Step 7: Prepare the full workflow data
            workflow_data = {
                "request_id": request_id,
                "input_type": input_type,
                "content": content,
                "routing": {
                    "agent_type": target_agent_type,
                    "capabilities": target_capabilities,
                    "confidence": routing_result.get("confidence", 1.0),
                    "reasoning": routing_result.get("reasoning", "")
                },
                "instructions": agent_instructions,
                "endpoint": self.agent_endpoint_map.get(target_agent_type, "/process"),
                "conversation_id": conversation_id,
                "context_id": context_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 8: Start a workflow through the orchestration manager
            orchestration_mgr = get_orchestration_manager()
            workflow_id, workflow_future = await orchestration_mgr.start_workflow(
                workflow_type="input_processing",
                initial_data=workflow_data,
                conversation_id=conversation_id,
                metadata={"request_id": request_id}
            )
            
            # Step 9: Update the context with the request and response
            if context_id:
                entity_data = {
                    "type": input_type,
                    "source": "input_agent",
                    "content": content,
                    "workflow_id": workflow_id,
                    "request_id": request_id,
                    "agent_type": target_agent_type
                }
                
                entity_id = self.context_manager.add_entity_to_context(
                    context_id,
                    entity_data,
                    input_type
                )
                
                if entity_id:
                    workflow_data["entity_id"] = entity_id
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            workflow_data["processing_time_ms"] = round(processing_time, 2)
            
            # Step 10: Return the response with all necessary information
            return {
                "success": True,
                "request_id": request_id,
                "workflow_id": workflow_id,
                "input_type": input_type,
                "agent_type": target_agent_type,
                "processing_time_ms": workflow_data["processing_time_ms"],
                "status": "processing",
                "instructions_summary": self._summarize_instructions(agent_instructions)
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "request_id": request_id
            }
    
    def _is_visualization_request(self, content: str) -> bool:
        """
        Check if the content is requesting a visualization or plot.
        
        Args:
            content: The request content
            
        Returns:
            Boolean indicating if this is a visualization request
        """
        visualization_keywords = [
            "plot", "chart", "graph", "visualize", "visualization", 
            "bar chart", "line graph", "scatter plot", "histogram", 
            "pie chart", "heatmap", "show me a", "create a chart"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in visualization_keywords)
    
    def _send_to_nlp_visualization(self, prompt: str) -> Dict[str, Any]:
        """
        Send the prompt to the NLP visualization endpoint.
        
        Args:
            prompt: The prompt text to send
            
        Returns:
            Response from the visualization endpoint
        """
        try:
            # Format the request exactly as: {"prompt": "..."}
            request_data = {
                "prompt": prompt
            }
            
            response = requests.post(
                self.nlp_visualization_endpoint,
                json=request_data,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Visualization endpoint returned error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Visualization service error: {response.status_code}",
                    "message": response.text
                }
                
        except Exception as e:
            logger.error(f"Failed to communicate with visualization endpoint: {str(e)}")
            return {
                "success": False,
                "error": f"Communication error: {str(e)}"
            }
    
    async def _generate_agent_instructions(self, 
                                         agent_type: str,
                                         request_data: Dict[str, Any],
                                         context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate detailed instructions for the target agent using LLM.
        
        Args:
            agent_type: The type of agent that will handle the request
            request_data: The processed request data
            context_data: Optional context data
            
        Returns:
            Detailed instructions for the agent
        """
        # Create a prompt for the LLM to generate detailed instructions
        prompt = self._create_instruction_prompt(agent_type, request_data, context_data)
        
        # Generate the instructions
        llm_result = self.llm_agent.generate_response(prompt, use_cot=True)
        
        if not llm_result.get("success", False):
            logger.error(f"Failed to generate agent instructions: {llm_result.get('error', 'Unknown error')}")
            return {
                "generic_instructions": f"Process this {agent_type} request.",
                "parameters": {},
                "step_by_step": ["Process the request"]
            }
        
        # Parse the instructions from the LLM output
        try:
            instructions = self._parse_agent_instructions(llm_result["response"], agent_type)
            return instructions
        except Exception as e:
            logger.error(f"Failed to parse agent instructions: {str(e)}")
            # Fallback to template-based instructions
            return self._create_template_instructions(agent_type, request_data)
    
    def _create_instruction_prompt(self, 
                                  agent_type: str,
                                  request_data: Dict[str, Any],
                                  context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for generating agent instructions.
        
        Args:
            agent_type: The type of agent that will handle the request
            request_data: The processed request data
            context_data: Optional context data
            
        Returns:
            Prompt string for the LLM
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append(f"""You are a specialist in creating detailed processing instructions for {agent_type} operations.
Your task is to generate specific, step-by-step instructions for a {agent_type} agent to process the user's request.
The instructions should be detailed, precise, and follow best practices for {agent_type} tasks.

Return your response in JSON format with the following structure:
{{
  "title": "Brief title describing the task",
  "description": "Detailed description of what needs to be done",
  "parameters": {{
    "param1": "value1",
    "param2": "value2",
    ...
  }},
  "step_by_step": [
    "Step 1: ...",
    "Step 2: ...",
    ...
  ],
  "expected_output": "Description of the expected output format"
}}""")
        
        # Add input information
        input_type = request_data.get("input_type", "unknown")
        prompt_parts.append(f"Input type: {input_type}")
        
        # Add content based on input type
        if input_type == "text":
            text = request_data.get("text", request_data.get("content", ""))
            prompt_parts.append(f"User request: {text}")
        elif input_type == "image":
            recognized_latex = request_data.get("recognized_latex", "")
            prompt_parts.append(f"Image content (recognized LaTeX): {recognized_latex}")
        
        # Add agent-specific capabilities
        agent_capabilities = self.agent_registry.get_agent_capabilities(f"{agent_type}_agent")
        if agent_capabilities:
            prompt_parts.append(f"\nAvailable {agent_type} capabilities:")
            for capability in agent_capabilities:
                prompt_parts.append(f"- {capability}")
        
        # Add context information if available
        if context_data:
            conversation_history = context_data.get("conversation_history", [])
            if conversation_history:
                prompt_parts.append("\nConversation history:")
                # Take the last 3 turns at most
                last_turns = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
                for turn in last_turns:
                    if "user" in turn:
                        prompt_parts.append(f"User: {turn['user']}")
                    if "assistant" in turn:
                        prompt_parts.append(f"Assistant: {turn['assistant']}")
        
        # Add agent-specific prompt sections
        if agent_type == "visualization":
            prompt_parts.append("""\nFor visualization requests, make sure to include:
1. The type of plot (bar chart, line graph, scatter plot, heatmap, etc.)
2. Data requirements and format
3. Axes labels and scales
4. Color schemes and styling
5. Any annotations or highlights
6. Interactive elements if needed""")
        elif agent_type == "math_computation":
            prompt_parts.append("""\nFor mathematical computation requests, make sure to include:
1. The precise mathematical operation to perform
2. Any variables or parameters with their domains
3. Computation steps in proper order
4. Expected output format (numeric, symbolic, LaTeX)
5. Verification steps if applicable""")
        
        # Final request
        prompt_parts.append("\nBased on the above information, generate detailed processing instructions in the specified JSON format.")
        
        return "\n\n".join(prompt_parts)
    
    def _parse_agent_instructions(self, llm_response: str, agent_type: str) -> Dict[str, Any]:
        """
        Parse the instructions from the LLM response.
        
        Args:
            llm_response: Response from the LLM
            agent_type: The type of agent that will handle the request
            
        Returns:
            Parsed instructions
        """
        # Extract JSON from response
        json_str = llm_response.strip()
        
        # Try to find JSON object if surrounded by other text
        if not json_str.startswith('{'):
            start_idx = json_str.find('{')
            if start_idx >= 0:
                end_idx = json_str.rfind('}')
                if end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx+1]
        
        try:
            instructions = json.loads(json_str)
            
            # Ensure all required fields are present
            required_fields = ["title", "description", "parameters", "step_by_step", "expected_output"]
            for field in required_fields:
                if field not in instructions:
                    if field == "parameters":
                        instructions[field] = {}
                    elif field == "step_by_step":
                        instructions[field] = ["Process the request"]
                    else:
                        instructions[field] = f"Missing {field}"
            
            return instructions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw response: {llm_response}")
            
            # Extract what might be useful from the text response
            lines = llm_response.split('\n')
            title = next((line for line in lines if line.strip()), "Processing request")
            
            # Extract potential steps (lines starting with numbers or "Step")
            steps = []
            for line in lines:
                line = line.strip()
                if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "Step")):
                    steps.append(line)
            
            if not steps:
                steps = ["Process the request"]
            
            return {
                "title": title,
                "description": llm_response[:200] + "..." if len(llm_response) > 200 else llm_response,
                "parameters": {},
                "step_by_step": steps,
                "expected_output": "Processed result"
            }
    
    def _create_template_instructions(self, agent_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create template-based instructions when LLM generation fails.
        
        Args:
            agent_type: The type of agent that will handle the request
            request_data: The processed request data
            
        Returns:
            Template-based instructions
        """
        if agent_type == "visualization":
            # Extract potential visualization parameters from the text
            content = request_data.get("text", request_data.get("content", ""))
            
            # Very simple extraction for demonstration purposes
            plot_type = "line chart"
            if "bar" in content.lower():
                plot_type = "bar chart"
            elif "scatter" in content.lower():
                plot_type = "scatter plot"
            elif "heat" in content.lower():
                plot_type = "heatmap"
            elif "pie" in content.lower():
                plot_type = "pie chart"
            
            template = self.agent_instruction_templates["visualization"]
            filled_template = template.format(
                plot_type=plot_type,
                title="Visualization from user request",
                data_description="Data derived from user request",
                x_axis="X-Axis",
                y_axis="Y-Axis",
                additional_instructions="Follow best practices for data visualization."
            )
            
            return {
                "title": f"Generate {plot_type}",
                "description": f"Create a {plot_type} based on the user's request",
                "parameters": {
                    "plot_type": plot_type,
                    "data_source": "user_request"
                },
                "step_by_step": filled_template.strip().split('\n'),
                "expected_output": "Visualization image and code"
            }
            
        elif agent_type == "math_computation":
            content = request_data.get("text", request_data.get("content", ""))
            
            template = self.agent_instruction_templates["math_computation"]
            filled_template = template.format(
                problem_statement=content
            )
            
            return {
                "title": "Solve mathematical problem",
                "description": f"Solve the mathematical problem: {content}",
                "parameters": {
                    "problem_type": "general",
                    "show_steps": True
                },
                "step_by_step": filled_template.strip().split('\n'),
                "expected_output": "Solution with steps in LaTeX format"
            }
            
        else:
            # Generic template for other agent types
            return {
                "title": f"Process {agent_type} request",
                "description": f"Process the user request for {agent_type}",
                "parameters": {},
                "step_by_step": [
                    f"1. Analyze the {agent_type} request",
                    "2. Process according to best practices",
                    "3. Return the result in appropriate format"
                ],
                "expected_output": "Processed result"
            }
    
    def _summarize_instructions(self, instructions: Dict[str, Any]) -> str:
        """
        Create a brief summary of the instructions for user feedback.
        
        Args:
            instructions: The detailed instructions
            
        Returns:
            Brief summary of the instructions
        """
        title = instructions.get("title", "Processing request")
        description = instructions.get("description", "")
        
        # Shorten description if needed
        if len(description) > 100:
            description = description[:97] + "..."
        
        steps = instructions.get("step_by_step", [])
        step_count = len(steps)
        
        return f"{title}: {description} ({step_count} steps)"

# Singleton instance for the input agent
_input_agent_instance = None

def get_input_agent() -> InputAgent:
    """Get or create the InputAgent singleton instance."""
    global _input_agent_instance
    if _input_agent_instance is None:
        _input_agent_instance = InputAgent()
    return _input_agent_instance 
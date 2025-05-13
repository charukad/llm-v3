"""
LLM Router Agent for intelligent multimodal request routing.

This agent uses the Core LLM to intelligently route requests to the appropriate 
specialized agents based on content analysis and understanding of the request.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import json
import uuid

from core.agent.llm_agent import CoreLLMAgent
from core.mistral.inference import InferenceEngine

logger = logging.getLogger(__name__)

class LLMRouterAgent:
    """
    LLM-powered Router Agent that intelligently routes requests to specialized agents.
    
    This agent serves as the central point of the system for processing all incoming 
    requests, using a large language model to analyze the content and determine which
    specialized agent is best suited to handle it.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM Router Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize the core LLM agent
        self.llm_agent = CoreLLMAgent(self.config.get("llm_config"))
        
        # Mapping of agent capabilities to agent types
        self.agent_capability_map = {
            "math_computation": ["algebraic_expression", "calculus", "equation_solving", "numerical_computation"],
            "ocr": ["handwriting_recognition", "math_symbol_recognition", "diagram_recognition"],
            "visualization": ["plot_generation", "graph_creation", "diagram_generation"],
            "text_processing": ["natural_language_understanding", "text_analysis", "semantic_parsing"],
            "search": ["information_retrieval", "knowledge_search"],
            "core_llm": ["question_answering", "explanation", "reasoning", "instruction_following"]
        }
        
        logger.info("Initialized LLM Router Agent")
    
    def route_request(self, processed_input: Dict[str, Any], 
                    context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the input and determine the optimal routing strategy.
        
        Args:
            processed_input: Dictionary containing processed input data
            context_data: Optional context data
            
        Returns:
            Dictionary containing routing decision and analysis
        """
        # Create a prompt for the LLM to analyze the request
        prompt = self._create_routing_prompt(processed_input, context_data)
        
        # Get routing analysis from LLM
        analysis_result = self.llm_agent.generate_response(prompt, use_cot=True)
        
        if not analysis_result.get("success", False):
            logger.error(f"Failed to generate routing analysis: {analysis_result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": f"Routing analysis failed: {analysis_result.get('error', 'Unknown error')}",
                "fallback_route": "core_llm"  # Default fallback
            }
        
        # Parse routing decision from LLM output
        try:
            routing_decision = self._parse_routing_decision(analysis_result["response"])
        except Exception as e:
            logger.error(f"Failed to parse routing decision: {str(e)}")
            routing_decision = {
                "primary_agent": "core_llm",
                "confidence": 0.7,
                "capabilities_needed": ["question_answering"],
                "reasoning": "Fallback due to parsing error"
            }
        
        # Create full routing response
        return {
            "success": True,
            "input_type": processed_input.get("input_type", "unknown"),
            "routing_id": str(uuid.uuid4()),
            "routing_decision": routing_decision,
            "processing_time_ms": analysis_result.get("processing_time_ms", 0),
            "original_input": processed_input
        }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from the message bus.
        
        Args:
            message: Message from the message bus
            
        Returns:
            Processing result with routing decision
        """
        # Extract message body
        body = message.get("body", {})
        
        # Extract processed input from message body
        processed_input = body.get("processed_input", body)
        context_data = body.get("context_data")
        
        # Route the request
        result = self.route_request(processed_input, context_data)
        
        # Add message metadata
        result["message_id"] = message.get("header", {}).get("message_id")
        result["message_type"] = message.get("header", {}).get("message_type")
        
        return result
    
    def _create_routing_prompt(self, processed_input: Dict[str, Any],
                             context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for the LLM to analyze and determine routing.
        
        Args:
            processed_input: Processed input data
            context_data: Optional context data
            
        Returns:
            Prompt string for the LLM
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are an intelligent router for a multimodal AI system. 
Your job is to analyze incoming requests and determine which specialized agent should handle them.
You must identify the primary capabilities needed to address the request.

Available agent types and their capabilities:
- math_computation: [algebraic_expression, calculus, equation_solving, numerical_computation]
- ocr: [handwriting_recognition, math_symbol_recognition, diagram_recognition]
- visualization: [plot_generation, graph_creation, diagram_generation]
- text_processing: [natural_language_understanding, text_analysis, semantic_parsing]
- search: [information_retrieval, knowledge_search]
- core_llm: [question_answering, explanation, reasoning, instruction_following]

Analyze the input carefully and return ONLY a JSON object with these fields:
{
  "primary_agent": "agent_type_name",
  "confidence": confidence_score_between_0_and_1,
  "capabilities_needed": ["capability1", "capability2"],
  "reasoning": "Brief explanation of why this agent was chosen"
}""")
        
        # Add input information
        input_type = processed_input.get("input_type", "unknown")
        prompt_parts.append(f"Input type: {input_type}")
        
        # Add content based on input type
        if input_type == "text":
            text = processed_input.get("text", processed_input.get("content", ""))
            prompt_parts.append(f"Text content: {text}")
            
        elif input_type == "image":
            recognized_latex = processed_input.get("recognized_latex", "")
            prompt_parts.append(f"Image content (recognized LaTeX): {recognized_latex}")
            
            # Add diagram information if available
            if "diagrams" in processed_input:
                diagrams = processed_input.get("diagrams", [])
                prompt_parts.append("Detected diagrams:")
                for i, diagram in enumerate(diagrams):
                    prompt_parts.append(f"- Diagram {i+1}: {diagram}")
                    
        elif input_type == "multipart":
            prompt_parts.append("Multipart content with multiple modalities:")
            parts = processed_input.get("parts", {})
            for key, part in parts.items():
                part_type = part.get("input_type", "unknown")
                prompt_parts.append(f"- Part '{key}' type: {part_type}")
                if part_type == "text":
                    prompt_parts.append(f"  Text: {part.get('text', '')}")
                elif part_type == "image":
                    prompt_parts.append(f"  Image (LaTeX): {part.get('recognized_latex', '')}")
        
        # Add context information if available
        if context_data:
            conversation_id = context_data.get("conversation_id")
            context_id = context_data.get("context_id")
            prompt_parts.append("\nContext information:")
            if conversation_id:
                prompt_parts.append(f"Conversation ID: {conversation_id}")
            if context_id:
                prompt_parts.append(f"Context ID: {context_id}")
            
            # Add conversation history summary if available
            conversation_history = context_data.get("conversation_history", [])
            if conversation_history:
                prompt_parts.append("\nConversation history summary:")
                history_summary = self._summarize_conversation_history(conversation_history)
                prompt_parts.append(history_summary)
        
        # Final request for routing decision
        prompt_parts.append("\nBased on the above input, determine the optimal agent for routing this request.")
        prompt_parts.append("Return ONLY the JSON object as specified in the instructions, with no additional text.")
        
        return "\n\n".join(prompt_parts)
    
    def _parse_routing_decision(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the routing decision from the LLM response.
        
        Args:
            llm_response: Response from the LLM
            
        Returns:
            Dictionary containing parsed routing decision
        """
        # Extract JSON from response (handle cases where LLM might add text before/after)
        json_str = llm_response.strip()
        
        # Try to find JSON object if surrounded by other text
        if not json_str.startswith('{'):
            start_idx = json_str.find('{')
            if start_idx >= 0:
                end_idx = json_str.rfind('}')
                if end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx+1]
        
        try:
            decision = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["primary_agent", "confidence", "capabilities_needed", "reasoning"]
            for field in required_fields:
                if field not in decision:
                    decision[field] = "core_llm" if field == "primary_agent" else (
                        0.7 if field == "confidence" else (
                            ["question_answering"] if field == "capabilities_needed" else "Missing field in response"
                        )
                    )
            
            # Validate agent type
            if decision["primary_agent"] not in self.agent_capability_map:
                logger.warning(f"Unknown agent type: {decision['primary_agent']}, falling back to core_llm")
                decision["primary_agent"] = "core_llm"
                decision["reasoning"] += " (corrected from unknown agent type)"
                
            # Ensure capabilities are valid for the agent
            valid_capabilities = self.agent_capability_map.get(decision["primary_agent"], [])
            validated_capabilities = []
            for cap in decision["capabilities_needed"]:
                if cap in valid_capabilities:
                    validated_capabilities.append(cap)
            
            # If no valid capabilities were found, use default ones for the agent
            if not validated_capabilities:
                validated_capabilities = valid_capabilities[:2] if valid_capabilities else ["question_answering"]
                
            decision["capabilities_needed"] = validated_capabilities
                
            return decision
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw response: {llm_response}")
            
            # Return default routing
            return {
                "primary_agent": "core_llm",
                "confidence": 0.7,
                "capabilities_needed": ["question_answering"],
                "reasoning": "Fallback due to JSON parsing error"
            }
    
    def _summarize_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Create a brief summary of the conversation history.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Brief summary of the conversation history
        """
        # For simplicity, we'll just take the last few turns
        last_turns = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        
        summary_lines = []
        for turn in last_turns:
            if "user" in turn:
                user_text = turn["user"]
                if len(user_text) > 100:
                    user_text = user_text[:97] + "..."
                summary_lines.append(f"User: {user_text}")
            if "assistant" in turn:
                assistant_text = turn["assistant"]
                if len(assistant_text) > 100:
                    assistant_text = assistant_text[:97] + "..."
                summary_lines.append(f"Assistant: {assistant_text}")
        
        if len(conversation_history) > 3:
            summary_lines.insert(0, f"[Conversation with {len(conversation_history)} total turns, showing last 3]")
            
        return "\n".join(summary_lines) 
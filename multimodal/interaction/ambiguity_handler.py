"""
Ambiguity handler for processing uncertain inputs and gathering clarification.

This module manages uncertain recognition results, generates clarification
requests, and processes user feedback.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AmbiguityHandler:
    """Handles ambiguity in input processing and user clarification."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ambiguity handler.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        self.max_suggestions = self.config.get("max_suggestions", 3)
        logger.info("Initialized ambiguity handler")
    
    def detect_ambiguities(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect ambiguities in processed input.
        
        Args:
            processed_input: Dictionary containing processed input data
            
        Returns:
            Dictionary with detected ambiguities
        """
        input_type = processed_input.get("input_type")
        
        if input_type == "image":
            return self._detect_image_ambiguities(processed_input)
        elif input_type == "text":
            return self._detect_text_ambiguities(processed_input)
        elif input_type == "multipart":
            return self._detect_multipart_ambiguities(processed_input)
        else:
            return {"has_ambiguities": False}
    
    def generate_clarification_request(self, ambiguities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a clarification request for ambiguous input.
        
        Args:
            ambiguities: Dictionary containing detected ambiguities
            
        Returns:
            Dictionary with clarification request
        """
        if not ambiguities.get("has_ambiguities", False):
            return {"needs_clarification": False}
        
        clarification_request = {
            "needs_clarification": True,
            "ambiguity_type": ambiguities["ambiguity_type"],
            "message": self._generate_clarification_message(ambiguities),
            "options": self._generate_clarification_options(ambiguities)
        }
        
        return clarification_request
    
    def process_clarification(self, original_input: Dict[str, Any], 
                            clarification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user clarification for ambiguous input.
        
        Args:
            original_input: Dictionary containing the original processed input
            clarification: Dictionary containing user clarification
            
        Returns:
            Updated processed input with clarification applied
        """
        input_type = original_input.get("input_type")
        
        if input_type == "image":
            return self._process_image_clarification(original_input, clarification)
        elif input_type == "text":
            return self._process_text_clarification(original_input, clarification)
        elif input_type == "multipart":
            return self._process_multipart_clarification(original_input, clarification)
        else:
            return original_input
    
    def _detect_image_ambiguities(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect ambiguities in image input.
        
        Args:
            image_data: Dictionary containing processed image data
            
        Returns:
            Dictionary with detected ambiguities
        """
        # Check overall confidence
        confidence = image_data.get("confidence", 0.0)
        recognized_latex = image_data.get("recognized_latex", "")
        
        if confidence < self.confidence_threshold:
            # Low overall confidence
            return {
                "has_ambiguities": True,
                "ambiguity_type": "low_confidence",
                "confidence": confidence,
                "recognized_latex": recognized_latex,
                "detected_symbols": image_data.get("symbols", [])
            }
        
        # Check for specific ambiguous symbols
        symbols = image_data.get("symbols", [])
        ambiguous_symbols = []
        
        for symbol in symbols:
            if symbol.get("confidence", 1.0) < self.confidence_threshold:
                ambiguous_symbols.append(symbol)
        
        if ambiguous_symbols:
            return {
                "has_ambiguities": True,
                "ambiguity_type": "ambiguous_symbols",
                "confidence": confidence,
                "recognized_latex": recognized_latex,
                "ambiguous_symbols": ambiguous_symbols
            }
        
        return {"has_ambiguities": False}
    
    def _detect_text_ambiguities(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect ambiguities in text input.
        
        Args:
            text_data: Dictionary containing processed text data
            
        Returns:
            Dictionary with detected ambiguities
        """
        # For text input, we're mostly concerned with ambiguous mathematical notation
        text_type = text_data.get("text_type", "")
        
        if text_type == "latex":
            # Check if the LaTeX is valid
            latex = text_data.get("latex", "")
            
            # In a real implementation, this would validate the LaTeX
            # For this example, we'll just check for unbalanced braces
            if self._has_unbalanced_braces(latex):
                return {
                    "has_ambiguities": True,
                    "ambiguity_type": "invalid_latex",
                    "latex": latex,
                    "issue": "unbalanced_braces"
                }
        
        return {"has_ambiguities": False}
    
    def _detect_multipart_ambiguities(self, multipart_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect ambiguities in multipart input.
        
        Args:
            multipart_data: Dictionary containing processed multipart data
            
        Returns:
            Dictionary with detected ambiguities
        """
        parts = multipart_data.get("parts", {})
        ambiguous_parts = {}
        
        for key, part_data in parts.items():
            input_type = part_data.get("input_type")
            
            if input_type == "image":
                ambiguities = self._detect_image_ambiguities(part_data)
                if ambiguities.get("has_ambiguities", False):
                    ambiguous_parts[key] = ambiguities
            elif input_type == "text":
                ambiguities = self._detect_text_ambiguities(part_data)
                if ambiguities.get("has_ambiguities", False):
                    ambiguous_parts[key] = ambiguities
        
        if ambiguous_parts:
            return {
                "has_ambiguities": True,
                "ambiguity_type": "multipart",
                "ambiguous_parts": ambiguous_parts
            }
        else:
            return {"has_ambiguities": False}
    
    def _generate_clarification_message(self, ambiguities: Dict[str, Any]) -> str:
        """
        Generate a clarification message based on detected ambiguities.
        
        Args:
            ambiguities: Dictionary containing detected ambiguities
            
        Returns:
            Clarification message string
        """
        ambiguity_type = ambiguities.get("ambiguity_type", "")
        
        if ambiguity_type == "low_confidence":
            return "I'm not entirely confident about my interpretation of your handwritten input. Could you confirm if this is correct?"
        
        elif ambiguity_type == "ambiguous_symbols":
            return "I'm uncertain about some symbols in your handwritten input. Could you clarify these specific parts?"
        
        elif ambiguity_type == "invalid_latex":
            return "There seems to be an issue with the LaTeX notation. Could you check if the expression is complete?"
        
        elif ambiguity_type == "multipart":
            return "I need clarification on some parts of your input. Could you help clarify them?"
        
        else:
            return "Could you please clarify your input? I'm not entirely sure about some parts."
    
    def _generate_clarification_options(self, ambiguities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate clarification options based on detected ambiguities.
        
        Args:
            ambiguities: Dictionary containing detected ambiguities
            
        Returns:
            List of clarification options
        """
        ambiguity_type = ambiguities.get("ambiguity_type", "")
        options = []
        
        if ambiguity_type == "low_confidence":
            recognized_latex = ambiguities.get("recognized_latex", "")
            
            options.append({
                "id": "confirm",
                "display": f"Confirm: {recognized_latex}",
                "value": {"action": "confirm", "latex": recognized_latex}
            })
            
            options.append({
                "id": "edit",
                "display": "Edit the interpretation",
                "value": {"action": "edit", "latex": recognized_latex}
            })
            
            options.append({
                "id": "retry",
                "display": "Upload a clearer image",
                "value": {"action": "retry"}
            })
        
        elif ambiguity_type == "ambiguous_symbols":
            recognized_latex = ambiguities.get("recognized_latex", "")
            ambiguous_symbols = ambiguities.get("ambiguous_symbols", [])
            
            options.append({
                "id": "confirm",
                "display": f"Confirm: {recognized_latex}",
                "value": {"action": "confirm", "latex": recognized_latex}
            })
            
            for i, symbol in enumerate(ambiguous_symbols[:self.max_suggestions]):
                options.append({
                    "id": f"symbol_{i}",
                    "display": f"Symbol at position {symbol.get('position', i)}: {symbol.get('text', '?')}",
                    "value": {"action": "edit_symbol", "symbol_id": i, "current": symbol.get('text', '')}
                })
        
        elif ambiguity_type == "invalid_latex":
            latex = ambiguities.get("latex", "")
            
            options.append({
                "id": "edit",
                "display": "Edit the LaTeX",
                "value": {"action": "edit", "latex": latex}
            })
        
        elif ambiguity_type == "multipart":
            ambiguous_parts = ambiguities.get("ambiguous_parts", {})
            
            for key, part_ambiguities in ambiguous_parts.items():
                part_type = part_ambiguities.get("ambiguity_type", "")
                
                if part_type == "low_confidence" or part_type == "ambiguous_symbols":
                    options.append({
                        "id": f"part_{key}",
                        "display": f"Clarify part {key}",
                        "value": {"action": "clarify_part", "part_id": key}
                    })
        
        # Always add an option to retype/redraw
        options.append({
            "id": "resubmit",
            "display": "Provide a new input",
            "value": {"action": "resubmit"}
        })
        
        return options
    
    def _process_image_clarification(self, original_input: Dict[str, Any],
                                  clarification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user clarification for image input.
        
        Args:
            original_input: Dictionary containing the original processed image
            clarification: Dictionary containing user clarification
            
        Returns:
            Updated processed input with clarification applied
        """
        action = clarification.get("action", "")
        
        if action == "confirm":
            # User confirmed the interpretation, just update confidence
            updated_input = original_input.copy()
            updated_input["confidence"] = 1.0
            updated_input["user_confirmed"] = True
            return updated_input
        
        elif action == "edit":
            # User provided a corrected LaTeX
            latex = clarification.get("latex", "")
            
            updated_input = original_input.copy()
            updated_input["recognized_latex"] = latex
            updated_input["confidence"] = 1.0
            updated_input["user_edited"] = True
            return updated_input
        
        elif action == "edit_symbol":
            # User corrected a specific symbol
            symbol_id = clarification.get("symbol_id", -1)
            new_text = clarification.get("new_text", "")
            
            if symbol_id >= 0 and symbol_id < len(original_input.get("symbols", [])):
                updated_input = original_input.copy()
                updated_input["symbols"] = original_input.get("symbols", []).copy()
                updated_input["symbols"][symbol_id]["text"] = new_text
                updated_input["symbols"][symbol_id]["confidence"] = 1.0
                updated_input["user_edited_symbol"] = True
                
                # Regenerate LaTeX with the corrected symbol
                # In a real implementation, this would call the LaTeX generator again
                # For this example, we'll just use the original LaTeX
                
                return updated_input
        
        # For other actions or if something went wrong, return the original input
        return original_input
    
    def _process_text_clarification(self, original_input: Dict[str, Any],
                                 clarification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user clarification for text input.
        
        Args:
            original_input: Dictionary containing the original processed text
            clarification: Dictionary containing user clarification
            
        Returns:
            Updated processed input with clarification applied
        """
        action = clarification.get("action", "")
        
        if action == "edit":
            # User provided a corrected LaTeX
            latex = clarification.get("latex", "")
            
            updated_input = original_input.copy()
            updated_input["latex"] = latex
            updated_input["user_edited"] = True
            return updated_input
        
        # For other actions or if something went wrong, return the original input
        return original_input
    
    def _process_multipart_clarification(self, original_input: Dict[str, Any],
                                      clarification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user clarification for multipart input.
        
        Args:
            original_input: Dictionary containing the original processed multipart input
            clarification: Dictionary containing user clarification
            
        Returns:
            Updated processed input with clarification applied
        """
        action = clarification.get("action", "")
        
        if action == "clarify_part":
            part_id = clarification.get("part_id", "")
            part_clarification = clarification.get("part_clarification", {})
            
            if part_id and part_id in original_input.get("parts", {}):
                updated_input = original_input.copy()
                updated_input["parts"] = original_input.get("parts", {}).copy()
                
                part_type = updated_input["parts"][part_id].get("input_type", "")
                
                if part_type == "image":
                    updated_input["parts"][part_id] = self._process_image_clarification(
                        updated_input["parts"][part_id], part_clarification
                    )
                elif part_type == "text":
                    updated_input["parts"][part_id] = self._process_text_clarification(
                        updated_input["parts"][part_id], part_clarification
                    )
                
                return updated_input
        
        # For other actions or if something went wrong, return the original input
        return original_input
    
    def _has_unbalanced_braces(self, latex: str) -> bool:
        """
        Check if a LaTeX string has unbalanced braces.
        
        Args:
            latex: LaTeX string to check
            
        Returns:
            True if the LaTeX has unbalanced braces, False otherwise
        """
        stack = []
        
        for char in latex:
            if char == '{':
                stack.append(char)
            elif char == '}':
                if not stack or stack[-1] != '{':
                    return True
                stack.pop()
        
        return len(stack) > 0

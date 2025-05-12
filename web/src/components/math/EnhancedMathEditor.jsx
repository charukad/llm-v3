import React, { useState, useEffect, useRef, useCallback } from 'react';
import MathKeyboard from './MathKeyboard';
import SymbolPalette from './SymbolPalette';
import DrawingCanvas from '../input/DrawingCanvas';
import LatexRenderer from '../latex/LatexRenderer';
import responseService from '../../services/responseService';
import preferencesService from '../../services/preferencesService';
import workflowService from '../../services/workflowService';
import './EnhancedMathEditor.css';

/**
 * Enhanced Math Editor Component
 * A sophisticated LaTeX editor with multiple input methods:
 * - Text-based LaTeX input with syntax highlighting
 * - Math keyboard with common symbols and structures
 * - Symbol palette organized by category
 * - Drawing canvas for handwritten input
 * 
 * @param {Object} props - Component props
 * @param {string} props.initialValue - Initial LaTeX value
 * @param {function} props.onChange - Callback when value changes
 * @param {function} props.onSubmit - Callback when editor is submitted
 * @param {boolean} props.showControls - Whether to show editor controls
 * @param {boolean} props.showPreview - Whether to show LaTeX preview
 * @param {string} props.placeholder - Placeholder text for the editor
 * @param {boolean} props.readOnly - Whether the editor is read-only
 */
const EnhancedMathEditor = ({
  initialValue = '',
  onChange = () => {},
  onSubmit = () => {},
  showControls = true,
  showPreview = true,
  placeholder = 'Enter a mathematical expression...',
  readOnly = false
}) => {
  // State for the editor
  const [value, setValue] = useState(initialValue);
  const [inputMethod, setInputMethod] = useState('text'); // 'text', 'keyboard', 'symbols', 'drawing'
  const [cursorPosition, setCursorPosition] = useState(0);
  const [history, setHistory] = useState([initialValue]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [isValid, setIsValid] = useState(true);
  const [validationMessage, setValidationMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Refs
  const textareaRef = useRef(null);
  const historyTimeout = useRef(null);
  
  // Load preferences
  const inputPrefs = preferencesService.get('input');
  
  // Update value when initialValue changes
  useEffect(() => {
    setValue(initialValue);
    // Reset history when initialValue changes externally
    setHistory([initialValue]);
    setHistoryIndex(0);
  }, [initialValue]);
  
  // Add to history when value changes (debounced)
  useEffect(() => {
    // Clear any existing timeout
    if (historyTimeout.current) {
      clearTimeout(historyTimeout.current);
    }
    
    // Only add to history if value is different from current history item
    if (value !== history[historyIndex]) {
      historyTimeout.current = setTimeout(() => {
        // Add new history item
        const newHistory = history.slice(0, historyIndex + 1).concat(value);
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);
      }, 1000); // 1 second debounce
    }
    
    // Call the onChange prop
    onChange(value);
    
    return () => {
      if (historyTimeout.current) {
        clearTimeout(historyTimeout.current);
      }
    };
  }, [value, history, historyIndex, onChange]);
  
  // Validate LaTeX expression
  const validateLatex = useCallback(async () => {
    if (!value.trim()) {
      setIsValid(true);
      setValidationMessage('');
      return true;
    }
    
    setIsProcessing(true);
    try {
      const response = await responseService.validateLatex(value);
      setIsValid(response.success);
      setValidationMessage(response.success ? '' : (response.error || 'Invalid LaTeX expression'));
      return response.success;
    } catch (error) {
      setIsValid(false);
      setValidationMessage('Error validating expression');
      return false;
    } finally {
      setIsProcessing(false);
    }
  }, [value]);
  
  // Handle inserting text at the cursor position
  const insertAtCursor = useCallback((text) => {
    if (readOnly) return;
    
    const newPosition = cursorPosition + text.length;
    const newValue = value.substring(0, cursorPosition) + text + value.substring(cursorPosition);
    setValue(newValue);
    setCursorPosition(newPosition);
    
    // Focus the textarea and set the cursor position
    if (textareaRef.current && inputMethod === 'text') {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = newPosition;
      textareaRef.current.selectionEnd = newPosition;
    }
  }, [value, cursorPosition, inputMethod, readOnly]);
  
  // Handle text input changes
  const handleTextChange = useCallback((e) => {
    if (readOnly) return;
    setValue(e.target.value);
    setCursorPosition(e.target.selectionStart);
  }, [readOnly]);
  
  // Handle cursor position changes
  const handleSelectionChange = useCallback((e) => {
    if (readOnly) return;
    setCursorPosition(e.target.selectionStart);
  }, [readOnly]);
  
  // Handle keyboard key insertion
  const handleKeyboardInsert = useCallback((key) => {
    insertAtCursor(key);
  }, [insertAtCursor]);
  
  // Handle symbol insertion
  const handleSymbolInsert = useCallback((symbol) => {
    insertAtCursor(symbol);
  }, [insertAtCursor]);
  
  // Handle drawing recognition
  const handleDrawingRecognized = useCallback(async (imageData) => {
    if (readOnly) return;
    
    setIsProcessing(true);
    try {
      // Call the handwriting recognition service
      const result = await workflowService.startHandwritingWorkflow(imageData);
      
      if (result.success && result.recognizedLatex) {
        // Insert the recognized LaTeX
        insertAtCursor(result.recognizedLatex);
      } else {
        console.error('Failed to recognize handwriting:', result.error);
      }
    } catch (error) {
      console.error('Error during handwriting recognition:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [insertAtCursor, readOnly]);
  
  // Handle undo
  const handleUndo = useCallback(() => {
    if (readOnly || historyIndex <= 0) return;
    
    const newIndex = historyIndex - 1;
    setHistoryIndex(newIndex);
    setValue(history[newIndex]);
  }, [history, historyIndex, readOnly]);
  
  // Handle redo
  const handleRedo = useCallback(() => {
    if (readOnly || historyIndex >= history.length - 1) return;
    
    const newIndex = historyIndex + 1;
    setHistoryIndex(newIndex);
    setValue(history[newIndex]);
  }, [history, historyIndex, readOnly]);
  
  // Handle submit
  const handleSubmit = useCallback(async () => {
    // Validate the expression first
    const isValidExpression = await validateLatex();
    
    if (isValidExpression) {
      onSubmit(value);
    }
  }, [value, validateLatex, onSubmit]);
  
  // Handle key down events for shortcuts
  const handleKeyDown = useCallback((e) => {
    // Shortcuts
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case 'z':
          e.preventDefault();
          if (e.shiftKey) {
            handleRedo();
          } else {
            handleUndo();
          }
          break;
        case 'y':
          e.preventDefault();
          handleRedo();
          break;
        case 'Enter':
          e.preventDefault();
          handleSubmit();
          break;
        default:
          break;
      }
    } else if (e.key === 'Enter' && e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleRedo, handleSubmit, handleUndo]);
  
  // Switch input method
  const switchInputMethod = (method) => {
    setInputMethod(method);
    
    // Focus the textarea when switching to text input
    if (method === 'text' && textareaRef.current) {
      setTimeout(() => {
        textareaRef.current.focus();
        textareaRef.current.selectionStart = cursorPosition;
        textareaRef.current.selectionEnd = cursorPosition;
      }, 0);
    }
  };
  
  // Convert natural language to LaTeX
  const handleNlToLatex = async () => {
    if (readOnly || !value.trim()) return;
    
    setIsProcessing(true);
    try {
      const response = await responseService.convertTextToLatex(value);
      
      if (response.success && response.data && response.data.latex) {
        setValue(response.data.latex);
      } else {
        console.error('Failed to convert to LaTeX:', response.error);
      }
    } catch (error) {
      console.error('Error converting to LaTeX:', error);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Render the editor
  return (
    <div className="enhanced-math-editor">
      {/* Input method tabs */}
      {showControls && (
        <div className="math-editor-tabs">
          <button 
            className={`tab-button ${inputMethod === 'text' ? 'active' : ''}`}
            onClick={() => switchInputMethod('text')}
            aria-label="Text input"
          >
            Text
          </button>
          <button 
            className={`tab-button ${inputMethod === 'keyboard' ? 'active' : ''}`}
            onClick={() => switchInputMethod('keyboard')}
            aria-label="Math keyboard"
          >
            Keyboard
          </button>
          <button 
            className={`tab-button ${inputMethod === 'symbols' ? 'active' : ''}`}
            onClick={() => switchInputMethod('symbols')}
            aria-label="Symbol palette"
          >
            Symbols
          </button>
          <button 
            className={`tab-button ${inputMethod === 'drawing' ? 'active' : ''}`}
            onClick={() => switchInputMethod('drawing')}
            aria-label="Drawing input"
          >
            Drawing
          </button>
        </div>
      )}
      
      {/* Input methods */}
      <div className="math-editor-content">
        {/* Text input */}
        {inputMethod === 'text' && (
          <div className="text-input-container">
            <textarea
              ref={textareaRef}
              className={`latex-textarea ${!isValid ? 'invalid' : ''}`}
              value={value}
              onChange={handleTextChange}
              onKeyDown={handleKeyDown}
              onClick={handleSelectionChange}
              onKeyUp={handleSelectionChange}
              placeholder={placeholder}
              disabled={readOnly || isProcessing}
              aria-label="LaTeX editor"
              aria-invalid={!isValid}
              aria-describedby="validation-message"
            />
            {!isValid && validationMessage && (
              <div id="validation-message" className="validation-message">
                {validationMessage}
              </div>
            )}
            {inputPrefs.autoSuggest && (
              <button 
                className="nl-to-latex-button"
                onClick={handleNlToLatex}
                disabled={readOnly || isProcessing || !value.trim()}
                aria-label="Convert to LaTeX"
                title="Convert natural language to LaTeX"
              >
                Convert to LaTeX
              </button>
            )}
          </div>
        )}
        
        {/* Math keyboard */}
        {inputMethod === 'keyboard' && (
          <MathKeyboard 
            onKeyPress={handleKeyboardInsert}
            disabled={readOnly || isProcessing}
          />
        )}
        
        {/* Symbol palette */}
        {inputMethod === 'symbols' && (
          <SymbolPalette 
            onSymbolSelect={handleSymbolInsert}
            disabled={readOnly || isProcessing}
          />
        )}
        
        {/* Drawing input */}
        {inputMethod === 'drawing' && (
          <DrawingCanvas 
            onRecognized={handleDrawingRecognized}
            disabled={readOnly || isProcessing}
          />
        )}
      </div>
      
      {/* Editor controls */}
      {showControls && (
        <div className="math-editor-controls">
          <button 
            className="control-button"
            onClick={handleUndo}
            disabled={readOnly || historyIndex <= 0 || isProcessing}
            aria-label="Undo"
            title="Undo (Ctrl+Z)"
          >
            Undo
          </button>
          <button 
            className="control-button"
            onClick={handleRedo}
            disabled={readOnly || historyIndex >= history.length - 1 || isProcessing}
            aria-label="Redo"
            title="Redo (Ctrl+Y)"
          >
            Redo
          </button>
          <button 
            className="control-button submit-button"
            onClick={handleSubmit}
            disabled={readOnly || !value.trim() || isProcessing}
            aria-label="Submit"
          >
            Submit
          </button>
        </div>
      )}
      
      {/* LaTeX preview */}
      {showPreview && value.trim() && (
        <div className="latex-preview">
          <div className="preview-label">Preview:</div>
          <LatexRenderer 
            latex={value}
            isBlock={true}
            className="preview-content"
          />
        </div>
      )}
      
      {/* Processing indicator */}
      {isProcessing && (
        <div className="processing-indicator" aria-live="polite">
          Processing...
        </div>
      )}
    </div>
  );
};

export default EnhancedMathEditor;

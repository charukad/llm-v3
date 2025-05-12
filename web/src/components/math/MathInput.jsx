import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import LatexRenderer from '../latex/LatexRenderer';
import LatexEditor from '../latex/LatexEditor';
import './MathInput.css';

/**
 * MathInput Component
 * 
 * A sophisticated input component for mathematical expressions that
 * supports natural language input, LaTeX editing, and real-time preview.
 * It converts natural language to LaTeX format using the API.
 */
const MathInput = ({
  initialValue = '',
  onSubmit = () => {},
  placeholder = 'Enter a mathematical expression...',
  renderEngine = 'mathjax',
  showLatexEditor = true,
  height = '200px',
  allowVoiceInput = false
}) => {
  const [inputMode, setInputMode] = useState('natural'); // 'natural' or 'latex'
  const [naturalInput, setNaturalInput] = useState(initialValue);
  const [latexOutput, setLatexOutput] = useState('');
  const [isConverting, setIsConverting] = useState(false);
  const [error, setError] = useState(null);
  const [showEditor, setShowEditor] = useState(false);
  const inputRef = useRef(null);
  
  // Convert natural language to LaTeX when input changes
  useEffect(() => {
    if (inputMode === 'natural' && naturalInput.trim()) {
      // Debounce conversion to avoid excessive API calls
      const timer = setTimeout(async () => {
        await convertToLatex();
      }, 800);
      
      return () => clearTimeout(timer);
    }
  }, [naturalInput, inputMode]);
  
  // Convert natural language to LaTeX using the API
  const convertToLatex = async () => {
    if (!naturalInput.trim()) {
      setLatexOutput('');
      return;
    }
    
    setIsConverting(true);
    setError(null);
    
    try {
      // Call the API to convert natural language to LaTeX
      const response = await fetch('/api/math/natural-to-latex', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          natural_text: naturalInput
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.latex) {
        setLatexOutput(data.latex);
      } else {
        throw new Error('No LaTeX output received from the API');
      }
    } catch (err) {
      console.error('Error converting to LaTeX:', err);
      setError(err.message || 'Failed to convert to LaTeX');
    } finally {
      setIsConverting(false);
    }
  };
  
  // Toggle between natural language and LaTeX input modes
  const toggleInputMode = () => {
    setInputMode(prev => prev === 'natural' ? 'latex' : 'natural');
    
    if (inputMode === 'latex') {
      // Switching to natural language mode
      setNaturalInput('');
      setLatexOutput('');
    } else {
      // Switching to LaTeX mode
      // Natural language input becomes LaTeX input
      setNaturalInput(latexOutput || '');
    }
  };
  
  // Handle input change based on current mode
  const handleInputChange = (e) => {
    const value = e.target.value;
    
    if (inputMode === 'natural') {
      setNaturalInput(value);
    } else {
      setNaturalInput(value);
      setLatexOutput(value);
    }
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e?.preventDefault();
    
    if (inputMode === 'natural' && !latexOutput) {
      // If in natural mode but no LaTeX yet, try to convert first
      convertToLatex().then(() => {
        onSubmit({ 
          naturalInput, 
          latexOutput: latexOutput || naturalInput 
        });
      });
    } else {
      onSubmit({ 
        naturalInput, 
        latexOutput: inputMode === 'natural' ? latexOutput : naturalInput 
      });
    }
  };
  
  // Toggle LaTeX editor visibility
  const toggleEditor = () => {
    setShowEditor(prev => !prev);
  };
  
  // Handle LaTeX editor changes
  const handleLatexEditorChange = (newLatex) => {
    if (inputMode === 'latex') {
      setNaturalInput(newLatex);
    }
    setLatexOutput(newLatex);
  };
  
  // Handle LaTeX editor save
  const handleLatexEditorSave = (latex) => {
    setLatexOutput(latex);
    setShowEditor(false);
    
    // Submit the form with the new LaTeX
    onSubmit({ 
      naturalInput, 
      latexOutput: latex 
    });
  };
  
  // Implement voice input if supported and enabled
  const startVoiceInput = () => {
    if (!allowVoiceInput || !('webkitSpeechRecognition' in window)) {
      return;
    }
    
    // This is a simple implementation - in a real app, you'd want to handle
    // more complex cases and browser compatibility
    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setNaturalInput(transcript);
    };
    
    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      setError(`Voice input error: ${event.error}`);
    };
    
    recognition.start();
  };
  
  return (
    <div className="math-input-container">
      <form onSubmit={handleSubmit} className="math-input-form">
        <div className="input-header">
          <div className="input-mode-toggle">
            <button
              type="button"
              className={`mode-button ${inputMode === 'natural' ? 'active' : ''}`}
              onClick={() => inputMode !== 'natural' && toggleInputMode()}
            >
              Natural Language
            </button>
            <button
              type="button"
              className={`mode-button ${inputMode === 'latex' ? 'active' : ''}`}
              onClick={() => inputMode !== 'latex' && toggleInputMode()}
            >
              LaTeX
            </button>
          </div>
          
          {showLatexEditor && (
            <button
              type="button"
              className="editor-toggle-button"
              onClick={toggleEditor}
              title={showEditor ? 'Hide LaTeX Editor' : 'Show LaTeX Editor'}
            >
              {showEditor ? 'Hide Editor' : 'Advanced Editor'}
            </button>
          )}
        </div>
        
        <div className="input-body">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              className="math-textarea"
              value={naturalInput}
              onChange={handleInputChange}
              placeholder={inputMode === 'natural' ? placeholder : 'Enter LaTeX code...'}
              style={{ height: height }}
            />
            
            {allowVoiceInput && 'webkitSpeechRecognition' in window && inputMode === 'natural' && (
              <button
                type="button"
                className="voice-input-button"
                onClick={startVoiceInput}
                title="Voice Input"
              >
                🎤
              </button>
            )}
          </div>
          
          {inputMode === 'natural' && (
            <div className="preview-wrapper">
              <div className="preview-header">Preview</div>
              <div className="latex-preview" style={{ minHeight: height }}>
                {isConverting ? (
                  <div className="converting-indicator">Converting...</div>
                ) : error ? (
                  <div className="error-message">{error}</div>
                ) : latexOutput ? (
                  <LatexRenderer
                    latex={latexOutput}
                    displayMode={true}
                    engine={renderEngine}
                  />
                ) : (
                  <div className="empty-preview">
                    Preview will appear here
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        <div className="input-footer">
          <button type="submit" className="submit-button">
            Submit
          </button>
        </div>
      </form>
      
      {showLatexEditor && showEditor && (
        <div className="latex-editor-wrapper">
          <LatexEditor
            initialValue={latexOutput || naturalInput}
            onChange={handleLatexEditorChange}
            onSave={handleLatexEditorSave}
            height="300px"
            renderEngine={renderEngine}
            displayMode={true}
            showSymbolToolbar={true}
            autoRender={true}
          />
        </div>
      )}
    </div>
  );
};

MathInput.propTypes = {
  initialValue: PropTypes.string,
  onSubmit: PropTypes.func,
  placeholder: PropTypes.string,
  renderEngine: PropTypes.oneOf(['mathjax', 'katex']),
  showLatexEditor: PropTypes.bool,
  height: PropTypes.string,
  allowVoiceInput: PropTypes.bool
};

export default MathInput;

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function MathInputPanel() {
  const [inputType, setInputType] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [contextId, setContextId] = useState('');
  const [workflow, setWorkflow] = useState(null);
  
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  
  // Initialize canvas for drawing
  useEffect(() => {
    if (inputType === 'drawing' && canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      context.lineWidth = 3;
      context.strokeStyle = '#000000';
      context.lineCap = 'round';
      context.lineJoin = 'round';
      
      // Clear canvas
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [inputType]);
  
  // Handle input type change
  const handleInputTypeChange = (type) => {
    setInputType(type);
    setError('');
    setResult(null);
  };
  
  // Handle text input change
  const handleTextInputChange = (e) => {
    setTextInput(e.target.value);
  };
  
  // Handle image file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  
  // Handle drawing
  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    context.beginPath();
    context.moveTo(x, y);
    setIsDrawing(true);
  };
  
  const draw = (e) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    context.lineTo(x, y);
    context.stroke();
  };
  
  const stopDrawing = () => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.closePath();
    setIsDrawing(false);
    
    // Convert canvas to file
    canvas.toBlob((blob) => {
      const file = new File([blob], 'drawing.png', { type: 'image/png' });
      setImageFile(file);
      setImagePreview(canvas.toDataURL());
    });
  };
  
  const clearDrawing = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    setImageFile(null);
    setImagePreview('');
  };
  
  // Send input to API
  const handleSubmit = async () => {
    setError('');
    setResult(null);
    setLoading(true);
    setWorkflow(null);
    
    try {
      let response;
      
      if (inputType === 'text') {
        // Submit text input
        if (!textInput.trim()) {
          throw new Error('Please enter some text');
        }
        
        response = await axios.post(`${API_BASE_URL}/workflow/process/text`, {
          text: textInput,
          context_id: contextId || undefined,
          conversation_id: conversationId || undefined
        });
      } else if (inputType === 'image' || inputType === 'drawing') {
        // Submit image or drawing
        if (!imageFile) {
          throw new Error('Please select an image or draw something');
        }
        
        const formData = new FormData();
        formData.append('file', imageFile);
        
        if (contextId) {
          formData.append('context_id', contextId);
        }
        
        if (conversationId) {
          formData.append('conversation_id', conversationId);
        }
        
        response = await axios.post(`${API_BASE_URL}/workflow/process/image`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
      }
      
      // Store workflow information
      setWorkflow(response.data);
      
      // Start polling for workflow result
      if (response.data.workflow_id) {
        pollWorkflowResult(response.data.workflow_id);
      }
      
      // Store context and conversation IDs
      if (response.data.context_id && !contextId) {
        setContextId(response.data.context_id);
      }
      
      if (response.data.conversation_id && !conversationId) {
        setConversationId(response.data.conversation_id);
      }
      
    } catch (err) {
      console.error('API error:', err);
      setError(`Error: ${err.message || 'Unknown error'}`);
      setLoading(false);
    }
  };
  
  // Poll for workflow result
  const pollWorkflowResult = async (workflowId) => {
    try {
      const poll = async () => {
        const response = await axios.get(`${API_BASE_URL}/workflow/result/${workflowId}`);
        
        if (response.data.success) {
          // Workflow completed
          setResult(response.data.result);
          setLoading(false);
          return true;
        } else if (response.data.state === 'error') {
          // Workflow errored
          throw new Error(response.data.message || 'Workflow failed');
        }
        
        // Workflow still in progress
        return false;
      };
      
      // Try polling with exponential backoff
      let attempts = 0;
      const maxAttempts = 10;
      
      const attemptPoll = async () => {
        attempts++;
        const completed = await poll();
        
        if (completed) {
          return;
        } else if (attempts >= maxAttempts) {
          throw new Error('Workflow took too long to complete');
        } else {
          // Wait with exponential backoff
          const delay = Math.min(1000 * Math.pow(1.5, attempts - 1), 10000);
          setTimeout(attemptPoll, delay);
        }
      };
      
      // Start polling
      attemptPoll();
      
    } catch (err) {
      console.error('Polling error:', err);
      setError(`Error: ${err.message || 'Unknown error'}`);
      setLoading(false);
    }
  };
  
  // Render math expressions using MathJax
  useEffect(() => {
    if (window.MathJax && result) {
      window.MathJax.typeset();
    }
  }, [result]);
  
  return (
    <div className="math-input-panel">
      <h2>Mathematical Multimodal LLM System</h2>
      
      {/* Input Type Selection */}
      <div className="input-type-selector">
        <button 
          className={inputType === 'text' ? 'active' : ''} 
          onClick={() => handleInputTypeChange('text')}
        >
          Text
        </button>
        <button 
          className={inputType === 'image' ? 'active' : ''} 
          onClick={() => handleInputTypeChange('image')}
        >
          Image
        </button>
        <button 
          className={inputType === 'drawing' ? 'active' : ''} 
          onClick={() => handleInputTypeChange('drawing')}
        >
          Draw
        </button>
      </div>
      
      {/* Text Input */}
      {inputType === 'text' && (
        <div className="text-input-container">
          <textarea
            value={textInput}
            onChange={handleTextInputChange}
            placeholder="Enter a mathematical question or expression..."
            rows={5}
          />
        </div>
      )}
      
      {/* Image Input */}
      {inputType === 'image' && (
        <div className="image-input-container">
          <input 
            type="file" 
            accept="image/*"
            onChange={handleFileChange}
            ref={fileInputRef}
            style={{ display: 'none' }}
          />
          <button onClick={() => fileInputRef.current.click()}>
            Select Image
          </button>
          {imagePreview && (
            <div className="image-preview">
              <img src={imagePreview} alt="Preview" />
            </div>
          )}
        </div>
      )}
      
      {/* Drawing Input */}
      {inputType === 'drawing' && (
        <div className="drawing-input-container">
          <div className="canvas-container">
            <canvas
              ref={canvasRef}
              width={500}
              height={300}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            />
          </div>
          <div className="drawing-controls">
            <button onClick={clearDrawing}>
              Clear
            </button>
          </div>
        </div>
      )}
      
      {/* Submit Button */}
      <div className="submit-container">
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? 'Processing...' : 'Process'}
        </button>
      </div>
      
      {/* Error Message */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      {/* Workflow Status */}
      {workflow && loading && (
        <div className="workflow-status">
          <div className="loading-spinner"></div>
          <div>Processing your request...</div>
        </div>
      )}
      
      {/* Result */}
      {result && (
        <div className="result-container">
          <h3>Result</h3>
          <div className="result-content">
            {result.response && (
              <div className="response-text" dangerouslySetInnerHTML={{ __html: result.response }}></div>
            )}
            {result.math_result && (
              <div className="math-result">
                <div className="math-latex">
                  <h4>LaTeX Result</h4>
                  <div className="latex-display">
                    <div dangerouslySetInnerHTML={{ __html: `\\[${result.math_result.latex_result}\\]` }}></div>
                  </div>
                </div>
                {result.math_result.steps && result.math_result.steps.length > 0 && (
                  <div className="step-by-step">
                    <h4>Step-by-Step Solution</h4>
                    <div className="steps">
                      {result.math_result.steps.map((step, index) => (
                        <div className="step" key={index}>
                          <div className="step-number">Step {index + 1}</div>
                          <div className="step-description">{step.description}</div>
                          {step.latex && (
                            <div className="step-latex" dangerouslySetInnerHTML={{ __html: `\\[${step.latex}\\]` }}></div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Session Info */}
      <div className="session-info">
        {contextId && <div>Context ID: {contextId}</div>}
        {conversationId && <div>Conversation ID: {conversationId}</div>}
        <button onClick={() => {
          setContextId('');
          setConversationId('');
        }}>
          Start New Session
        </button>
      </div>
    </div>
  );
}

export default MathInputPanel;

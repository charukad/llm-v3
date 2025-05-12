import React, { useState, useCallback, useEffect } from 'react';
import EnhancedMathEditor from './EnhancedMathEditor';
import CameraCapture from '../input/CameraCapture';
import ResponsePreferences from '../preferences/ResponsePreferences';
import workflowService from '../../services/workflowService';
import preferencesService from '../../services/preferencesService';
import './MathQueryPanel.css';

/**
 * Math Query Panel Component
 * The main interface for users to interact with the system
 * 
 * @param {Object} props - Component props
 * @param {function} props.onSubmit - Callback when query is submitted
 * @param {function} props.onResponse - Callback when response is received
 * @param {function} props.onError - Callback when error occurs
 * @param {boolean} props.showPreferences - Whether to show response preferences
 */
const MathQueryPanel = ({
  onSubmit = () => {},
  onResponse = () => {},
  onError = () => {},
  showPreferences = true
}) => {
  // State for the component
  const [inputMethod, setInputMethod] = useState('text'); // 'text', 'latex', 'camera'
  const [textQuery, setTextQuery] = useState('');
  const [latexQuery, setLatexQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [examples, setExamples] = useState([
    'Find the derivative of f(x) = x² sin(x)',
    'Solve the equation 2x + 5 = 3x - 7',
    'Calculate the integral of ln(x) from 1 to 3',
    'Find the eigenvalues of [[1,2],[3,4]]'
  ]);
  
  // Load preferences for default input method
  useEffect(() => {
    const defaultMethod = preferencesService.getValue('input', 'defaultInputMethod');
    if (defaultMethod) {
      setInputMethod(defaultMethod === 'drawing' ? 'latex' : defaultMethod);
    }
  }, []);
  
  // Handle text input changes
  const handleTextChange = useCallback((e) => {
    setTextQuery(e.target.value);
    setError('');
  }, []);
  
  // Handle LaTeX editor changes
  const handleLatexChange = useCallback((latex) => {
    setLatexQuery(latex);
    setError('');
  }, []);
  
  // Handle camera capture
  const handleCameraCapture = useCallback(async (imageData) => {
    setIsLoading(true);
    setError('');
    
    try {
      // Start a handwriting recognition workflow
      const result = await workflowService.startHandwritingWorkflow(imageData);
      
      if (result.success && result.recognizedLatex) {
        // Set the recognized LaTeX
        setLatexQuery(result.recognizedLatex);
        // Switch to the LaTeX input method
        setInputMethod('latex');
      } else {
        setError(result.error || 'Failed to recognize handwriting');
      }
    } catch (error) {
      setError(error.message || 'An error occurred while processing the image');
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Handle form submission
  const handleSubmit = useCallback(async (e) => {
    if (e) e.preventDefault();
    
    // Get the query based on input method
    const query = inputMethod === 'text' ? textQuery : latexQuery;
    
    // Validate input
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    // Call the onSubmit callback
    onSubmit({
      query,
      inputMethod,
      timestamp: new Date().toISOString()
    });
    
    try {
      // Get response preferences
      const preferences = preferencesService.get('response');
      
      // Execute the workflow
      const workflowType = 'math';
      const result = await workflowService.executeWorkflow(workflowType, query, {
        showStepByStep: preferences.showStepByStep,
        showVisualizations: preferences.showVisualizations,
        detailLevel: preferences.detailLevel
      });
      
      if (result.success) {
        // Call the onResponse callback
        onResponse(result.data);
      } else {
        setError(result.error || 'Failed to process query');
        onError(result.error || 'Failed to process query');
      }
    } catch (error) {
      const errorMessage = error.message || 'An error occurred while processing the query';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [inputMethod, textQuery, latexQuery, onSubmit, onResponse, onError]);
  
  // Handle example click
  const handleExampleClick = useCallback((example) => {
    setTextQuery(example);
    setInputMethod('text');
    setError('');
  }, []);
  
  // Render the component
  return (
    <div className="math-query-panel">
      {/* Input tabs */}
      <div className="query-tabs">
        <button 
          className={`tab-button ${inputMethod === 'text' ? 'active' : ''}`}
          onClick={() => setInputMethod('text')}
          disabled={isLoading}
          aria-label="Text input"
        >
          Text
        </button>
        <button 
          className={`tab-button ${inputMethod === 'latex' ? 'active' : ''}`}
          onClick={() => setInputMethod('latex')}
          disabled={isLoading}
          aria-label="LaTeX input"
        >
          LaTeX
        </button>
        <button 
          className={`tab-button ${inputMethod === 'camera' ? 'active' : ''}`}
          onClick={() => setInputMethod('camera')}
          disabled={isLoading}
          aria-label="Camera input"
        >
          Camera
        </button>
      </div>
      
      {/* Query form */}
      <form className="query-form" onSubmit={handleSubmit}>
        <div className="query-input-container">
          {/* Text input */}
          {inputMethod === 'text' && (
            <textarea
              className="text-query-input"
              value={textQuery}
              onChange={handleTextChange}
              placeholder="Enter your mathematical question..."
              disabled={isLoading}
              aria-label="Mathematical question"
            />
          )}
          
          {/* LaTeX input */}
          {inputMethod === 'latex' && (
            <EnhancedMathEditor
              initialValue={latexQuery}
              onChange={handleLatexChange}
              onSubmit={handleSubmit}
              showControls={true}
              showPreview={true}
              placeholder="Enter a LaTeX expression..."
              readOnly={isLoading}
            />
          )}
          
          {/* Camera input */}
          {inputMethod === 'camera' && (
            <CameraCapture
              onCapture={handleCameraCapture}
              disabled={isLoading}
            />
          )}
        </div>
        
        {/* Error message */}
        {error && (
          <div className="error-message" aria-live="assertive">
            {error}
          </div>
        )}
        
        {/* Submit button */}
        {(inputMethod === 'text' || (inputMethod === 'latex' && !document.querySelector('.enhanced-math-editor .submit-button'))) && (
          <button 
            type="submit" 
            className="submit-button"
            disabled={isLoading || (inputMethod === 'text' && !textQuery.trim()) || (inputMethod === 'latex' && !latexQuery.trim())}
            aria-label="Submit query"
          >
            {isLoading ? 'Processing...' : 'Submit'}
          </button>
        )}
      </form>
      
      {/* Response preferences */}
      {showPreferences && (
        <ResponsePreferences />
      )}
      
      {/* Example queries */}
      <div className="example-queries">
        <h3>Example queries:</h3>
        <ul>
          {examples.map((example, index) => (
            <li key={index}>
              <button
                className="example-button"
                onClick={() => handleExampleClick(example)}
                disabled={isLoading}
                aria-label={`Example: ${example}`}
              >
                {example}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default MathQueryPanel;

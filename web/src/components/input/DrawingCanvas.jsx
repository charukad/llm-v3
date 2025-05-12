/**
 * Drawing Canvas Component
 * 
 * A canvas for capturing handwritten mathematical notation
 * that integrates with handwriting recognition services.
 */

import React, { useRef, useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import './DrawingCanvas.css';

const DrawingCanvas = forwardRef(({
  width = 600,
  height = 400,
  lineWidth = 3,
  lineColor = '#000',
  backgroundColor = '#fff',
  onRecognized,
  recognitionEndpoint = '/api/handwriting-recognition',
}, ref) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasContent, setHasContent] = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [error, setError] = useState(null);
  
  // Drawing context
  const contextRef = useRef(null);
  
  // Initialize canvas on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = width * 2; // For better resolution on high DPI screens
    canvas.height = height * 2;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    
    const context = canvas.getContext('2d');
    context.scale(2, 2); // Scale for high DPI
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.lineWidth = lineWidth;
    context.strokeStyle = lineColor;
    
    // Fill with background color
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    contextRef.current = context;
  }, [width, height, lineWidth, lineColor, backgroundColor]);
  
  // Start drawing
  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.beginPath();
    contextRef.current.moveTo(offsetX, offsetY);
    setIsDrawing(true);
    setHasContent(true);
  };
  
  // Draw
  const draw = ({ nativeEvent }) => {
    if (!isDrawing) return;
    
    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.lineTo(offsetX, offsetY);
    contextRef.current.stroke();
  };
  
  // Stop drawing
  const stopDrawing = () => {
    contextRef.current.closePath();
    setIsDrawing(false);
  };
  
  // Handle touch events for mobile
  const getTouchPos = (canvas, touchEvent) => {
    const rect = canvas.getBoundingClientRect();
    return {
      offsetX: touchEvent.touches[0].clientX - rect.left,
      offsetY: touchEvent.touches[0].clientY - rect.top
    };
  };
  
  const handleTouchStart = (e) => {
    e.preventDefault();
    const touchPos = getTouchPos(canvasRef.current, e);
    startDrawing({ nativeEvent: touchPos });
  };
  
  const handleTouchMove = (e) => {
    e.preventDefault();
    const touchPos = getTouchPos(canvasRef.current, e);
    draw({ nativeEvent: touchPos });
  };
  
  const handleTouchEnd = (e) => {
    e.preventDefault();
    stopDrawing();
  };
  
  // Clear canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width / 2, canvas.height / 2); // Adjust for the scale
    setHasContent(false);
    setError(null);
  };
  
  // Recognize handwritten math
  const recognizeHandwriting = async () => {
    if (!hasContent) return;
    
    setIsRecognizing(true);
    setError(null);
    
    try {
      // Convert canvas to blob
      const canvas = canvasRef.current;
      const blob = await new Promise(resolve => {
        canvas.toBlob(resolve, 'image/png');
      });
      
      // Create form data
      const formData = new FormData();
      formData.append('image', blob, 'math-expression.png');
      
      // Send to recognition endpoint
      const response = await fetch(recognitionEndpoint, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Recognition failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Call onRecognized callback with result
      if (onRecognized && result.latex) {
        onRecognized(result.latex);
      } else {
        throw new Error('No LaTeX expression found in recognition result');
      }
    } catch (err) {
      console.error('Handwriting recognition error:', err);
      setError(`Recognition failed: ${err.message}`);
    } finally {
      setIsRecognizing(false);
    }
  };
  
  // Recognize handwriting using a mock for testing/development
  const recognizeHandwritingMock = () => {
    if (!hasContent) return;
    
    setIsRecognizing(true);
    setError(null);
    
    // Simulate server delay
    setTimeout(() => {
      // Mock recognition result - random simple expressions
      const mockExpressions = [
        'x^2 + 2x + 1',
        '\\frac{1}{2}',
        '\\int x^2 dx',
        '\\sum_{i=1}^{n} i',
        'a + b = c',
        '\\sqrt{x^2 + y^2}',
        'e^{i\\pi} + 1 = 0',
        '\\frac{d}{dx}(x^2) = 2x',
      ];
      
      const randomIndex = Math.floor(Math.random() * mockExpressions.length);
      const result = { latex: mockExpressions[randomIndex] };
      
      // Call onRecognized callback with result
      if (onRecognized) {
        onRecognized(result.latex);
      }
      
      setIsRecognizing(false);
    }, 1000); // Simulate 1 second delay
  };
  
  // Expose methods via ref
  useImperativeHandle(ref, () => ({
    clear: clearCanvas,
    recognize: recognizeHandwriting,
    getCanvas: () => canvasRef.current,
  }));
  
  return (
    <div className="drawing-canvas-container">
      <div className="canvas-wrapper">
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          className="drawing-canvas"
          aria-label="Drawing canvas for handwritten mathematical expressions"
        />
      </div>
      
      <div className="canvas-controls">
        <button
          type="button"
          onClick={clearCanvas}
          className="canvas-button"
          disabled={!hasContent || isRecognizing}
          aria-label="Clear canvas"
        >
          Clear
        </button>
        
        <button
          type="button"
          onClick={recognizeHandwritingMock} // Use mock for development
          className="canvas-button recognize-button"
          disabled={!hasContent || isRecognizing}
          aria-label="Recognize handwriting"
        >
          {isRecognizing ? 'Recognizing...' : 'Recognize'}
        </button>
      </div>
      
      {error && (
        <div className="canvas-error" role="alert">
          {error}
        </div>
      )}
      
      <div className="canvas-instructions">
        <p>Write a mathematical expression using your mouse or touch input.</p>
        <p>Press "Recognize" when finished to convert to LaTeX.</p>
      </div>
    </div>
  );
});

DrawingCanvas.displayName = 'DrawingCanvas';

export default DrawingCanvas;

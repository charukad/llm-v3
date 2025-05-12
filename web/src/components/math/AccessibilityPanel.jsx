/**
 * AccessibilityPanel Component
 * 
 * Provides accessibility features for mathematical content, including
 * screen reader descriptions and keyboard navigation controls.
 */

import React, { useState } from 'react';
import './AccessibilityPanel.css';

const AccessibilityPanel = ({ latexExpression, description }) => {
  const [showPanel, setShowPanel] = useState(false);
  const [fontSize, setFontSize] = useState(16);
  const [highContrast, setHighContrast] = useState(false);
  
  // Toggle panel visibility
  const togglePanel = () => {
    setShowPanel(!showPanel);
  };
  
  // Increase font size
  const increaseFontSize = () => {
    setFontSize(Math.min(fontSize + 2, 24));
  };
  
  // Decrease font size
  const decreaseFontSize = () => {
    setFontSize(Math.max(fontSize - 2, 12));
  };
  
  // Toggle high contrast mode
  const toggleHighContrast = () => {
    setHighContrast(!highContrast);
  };
  
  // Read description aloud using browser's speech synthesis
  const readAloud = () => {
    if (!description) return;
    
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(description);
      speechSynthesis.speak(utterance);
    }
  };
  
  return (
    <div className={`accessibility-panel ${highContrast ? 'high-contrast' : ''}`}>
      <button 
        type="button"
        onClick={togglePanel}
        className="accessibility-toggle"
        aria-expanded={showPanel}
        aria-label="Toggle accessibility panel"
      >
        <span className="accessibility-icon">♿</span> 
        {showPanel ? 'Hide Accessibility Options' : 'Show Accessibility Options'}
      </button>
      
      {showPanel && (
        <div className="accessibility-content">
          <div className="accessibility-controls">
            <div className="control-group">
              <label htmlFor="font-size">Text Size:</label>
              <div className="button-group">
                <button 
                  type="button" 
                  onClick={decreaseFontSize}
                  aria-label="Decrease font size"
                  disabled={fontSize <= 12}
                >
                  A-
                </button>
                <span id="font-size">{fontSize}px</span>
                <button 
                  type="button" 
                  onClick={increaseFontSize}
                  aria-label="Increase font size"
                  disabled={fontSize >= 24}
                >
                  A+
                </button>
              </div>
            </div>
            
            <div className="control-group">
              <label htmlFor="high-contrast">High Contrast:</label>
              <div className="button-group">
                <button 
                  type="button" 
                  onClick={toggleHighContrast}
                  className={highContrast ? 'active' : ''}
                  aria-pressed={highContrast}
                  id="high-contrast"
                >
                  {highContrast ? 'On' : 'Off'}
                </button>
              </div>
            </div>
            
            <div className="control-group">
              <label>Text-to-Speech:</label>
              <div className="button-group">
                <button 
                  type="button" 
                  onClick={readAloud}
                  aria-label="Read expression aloud"
                  disabled={!description}
                >
                  Read Aloud
                </button>
              </div>
            </div>
          </div>
          
          <div className="accessibility-description" style={{ fontSize: `${fontSize}px` }}>
            <h4>LaTeX Expression:</h4>
            <pre>{latexExpression || 'No expression available'}</pre>
            
            <h4>Spoken Description:</h4>
            <p>{description || 'No description available'}</p>
          </div>
          
          <div className="keyboard-shortcuts">
            <h4>Keyboard Shortcuts:</h4>
            <ul>
              <li><kbd>Tab</kbd> - Navigate between controls</li>
              <li><kbd>Space</kbd> or <kbd>Enter</kbd> - Activate buttons</li>
              <li><kbd>Esc</kbd> - Close panels</li>
              <li><kbd>Ctrl</kbd> + <kbd>Z</kbd> - Undo</li>
              <li><kbd>Ctrl</kbd> + <kbd>Y</kbd> - Redo</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default AccessibilityPanel;

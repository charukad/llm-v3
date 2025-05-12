import React, { useState, useEffect } from 'react';
import preferencesService from '../../services/preferencesService';
import './ResponsePreferences.css';

/**
 * ResponsePreferences Component
 * Allows users to configure preferences for mathematical responses
 * 
 * @param {Object} props - Component props
 * @param {function} props.onChange - Callback when preferences change
 * @param {boolean} props.compact - Whether to display in compact mode
 * @param {string} props.className - Additional CSS class
 */
const ResponsePreferences = ({
  onChange = () => {},
  compact = false,
  className = ''
}) => {
  // State for preferences
  const [preferences, setPreferences] = useState({
    showStepByStep: true,
    showVisualizations: true,
    detailLevel: 'medium',
    preferredFormat: 'combined'
  });
  
  // State for UI
  const [isExpanded, setIsExpanded] = useState(!compact);
  
  // Load preferences from service
  useEffect(() => {
    const responsePrefs = preferencesService.get('response');
    if (responsePrefs) {
      setPreferences(responsePrefs);
    }
  }, []);
  
  // Handle preference changes
  const handlePreferenceChange = (key, value) => {
    const updatedPreferences = {
      ...preferences,
      [key]: value
    };
    
    setPreferences(updatedPreferences);
    
    // Update the preferences service
    preferencesService.update('response', updatedPreferences);
    
    // Call the onChange callback
    onChange(updatedPreferences);
  };
  
  // Toggle expanded state
  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };
  
  // Reset preferences to defaults
  const resetPreferences = () => {
    preferencesService.resetCategory('response');
    const defaultPrefs = preferencesService.get('response');
    setPreferences(defaultPrefs);
    onChange(defaultPrefs);
  };
  
  return (
    <div className={`response-preferences ${compact ? 'compact' : ''} ${className}`}>
      {/* Header with toggle */}
      <div className="preferences-header" onClick={toggleExpanded}>
        <h3>Response Preferences</h3>
        <button 
          className="toggle-button"
          aria-label={isExpanded ? 'Collapse preferences' : 'Expand preferences'}
          aria-expanded={isExpanded}
        >
          {isExpanded ? '▼' : '►'}
        </button>
      </div>
      
      {/* Preferences content */}
      {isExpanded && (
        <div className="preferences-content">
          {/* Step-by-step toggle */}
          <div className="preference-item">
            <label className="toggle-label">
              <input 
                type="checkbox"
                checked={preferences.showStepByStep}
                onChange={(e) => handlePreferenceChange('showStepByStep', e.target.checked)}
                aria-label="Show step-by-step solution"
              />
              <span className="toggle-text">Show Step-by-Step Solution</span>
            </label>
          </div>
          
          {/* Visualizations toggle */}
          <div className="preference-item">
            <label className="toggle-label">
              <input 
                type="checkbox"
                checked={preferences.showVisualizations}
                onChange={(e) => handlePreferenceChange('showVisualizations', e.target.checked)}
                aria-label="Show visualizations"
              />
              <span className="toggle-text">Show Visualizations</span>
            </label>
          </div>
          
          {/* Detail level select */}
          <div className="preference-item">
            <label className="select-label">
              <span className="select-text">Detail Level:</span>
              <select 
                value={preferences.detailLevel}
                onChange={(e) => handlePreferenceChange('detailLevel', e.target.value)}
                aria-label="Detail level"
              >
                <option value="basic">Basic</option>
                <option value="medium">Medium</option>
                <option value="detailed">Detailed</option>
              </select>
            </label>
          </div>
          
          {/* Preferred format select */}
          <div className="preference-item">
            <label className="select-label">
              <span className="select-text">Preferred Format:</span>
              <select 
                value={preferences.preferredFormat}
                onChange={(e) => handlePreferenceChange('preferredFormat', e.target.value)}
                aria-label="Preferred format"
              >
                <option value="text">Text Only</option>
                <option value="latex">LaTeX Focus</option>
                <option value="combined">Combined</option>
              </select>
            </label>
          </div>
          
          {/* Reset button */}
          <div className="preference-actions">
            <button 
              className="reset-button"
              onClick={resetPreferences}
              aria-label="Reset preferences to defaults"
            >
              Reset to Defaults
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResponsePreferences;

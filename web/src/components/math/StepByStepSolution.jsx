import React, { useState, useEffect } from 'react';
import LatexRenderer from '../latex/LatexRenderer';
import preferencesService from '../../services/preferencesService';
import './StepByStepSolution.css';

/**
 * StepByStepSolution Component
 * Displays a step-by-step mathematical solution with explanations
 * 
 * @param {Object} props - Component props
 * @param {Array} props.steps - The solution steps to display
 * @param {string} props.detailLevel - The level of detail to show ('basic', 'medium', 'detailed')
 * @param {boolean} props.expandAll - Whether to expand all steps initially
 * @param {string} props.className - Additional CSS class
 */
const StepByStepSolution = ({
  steps = [],
  detailLevel = 'medium',
  expandAll = false,
  className = ''
}) => {
  // State for tracking expanded steps
  const [expandedSteps, setExpandedSteps] = useState({});
  
  // Initialize expanded state from preferences or props
  useEffect(() => {
    // If expandAll prop is true, expand all steps
    if (expandAll) {
      const expanded = {};
      steps.forEach(step => {
        expanded[step.id || step.number] = true;
      });
      setExpandedSteps(expanded);
      return;
    }
    
    // Otherwise, expand key steps based on detail level
    const expanded = {};
    steps.forEach(step => {
      const stepId = step.id || step.number;
      
      // Determine if step should be expanded based on detail level
      if (detailLevel === 'detailed') {
        // Expand all steps for detailed view
        expanded[stepId] = true;
      } else if (detailLevel === 'medium') {
        // Expand key steps for medium view
        expanded[stepId] = step.isKeyStep;
      } else if (detailLevel === 'basic') {
        // Only expand the first and last steps, and very key steps for basic view
        expanded[stepId] = 
          step.number === 1 || 
          step.number === steps.length || 
          (step.isKeyStep && step.type === 'major');
      }
    });
    
    setExpandedSteps(expanded);
  }, [steps, detailLevel, expandAll]);
  
  // Filter steps based on detail level
  const filteredSteps = steps.filter(step => {
    if (detailLevel === 'detailed') {
      // Show all steps
      return true;
    } else if (detailLevel === 'medium') {
      // Skip minor steps unless they are key steps
      return step.type !== 'minor' || step.isKeyStep;
    } else if (detailLevel === 'basic') {
      // Only show key steps or major transitions
      return step.isKeyStep || step.type === 'major' || 
        step.number === 1 || step.number === steps.length;
    }
    
    // Default to showing the step
    return true;
  });
  
  // Toggle step expansion
  const toggleStep = (stepId) => {
    setExpandedSteps(prev => ({
      ...prev,
      [stepId]: !prev[stepId]
    }));
  };
  
  // Expand all steps
  const expandAllSteps = () => {
    const expanded = {};
    steps.forEach(step => {
      expanded[step.id || step.number] = true;
    });
    setExpandedSteps(expanded);
  };
  
  // Collapse all steps
  const collapseAllSteps = () => {
    setExpandedSteps({});
  };
  
  // If there are no steps, show a message
  if (!steps || steps.length === 0) {
    return (
      <div className={`step-by-step-solution empty ${className}`}>
        <p>No solution steps available.</p>
      </div>
    );
  }
  
  return (
    <div className={`step-by-step-solution ${className}`}>
      {/* Controls */}
      <div className="solution-controls">
        <button 
          onClick={expandAllSteps}
          className="control-button"
          aria-label="Expand all steps"
        >
          Expand All
        </button>
        <button 
          onClick={collapseAllSteps}
          className="control-button"
          aria-label="Collapse all steps"
        >
          Collapse All
        </button>
        
        {/* Detail level controls if not provided by the parent */}
        {!detailLevel && (
          <div className="detail-level-controls">
            <span>Detail Level:</span>
            <select 
              value={detailLevel} 
              onChange={(e) => {
                const newLevel = e.target.value;
                // Store the preference
                preferencesService.setValue('response', 'detailLevel', newLevel);
              }}
              aria-label="Solution detail level"
            >
              <option value="basic">Basic</option>
              <option value="medium">Medium</option>
              <option value="detailed">Detailed</option>
            </select>
          </div>
        )}
      </div>
      
      {/* Steps list */}
      <div className="steps-list">
        {filteredSteps.map((step) => {
          const stepId = step.id || step.number;
          const isExpanded = expandedSteps[stepId] || false;
          
          return (
            <div 
              key={stepId}
              className={`solution-step ${step.type || 'general'} ${step.isKeyStep ? 'key-step' : ''} ${isExpanded ? 'expanded' : 'collapsed'}`}
            >
              <div 
                className="step-header" 
                onClick={() => toggleStep(stepId)}
                aria-expanded={isExpanded}
                aria-controls={`step-content-${stepId}`}
              >
                <div className="step-indicator">
                  <span className="step-number">{step.number}</span>
                  {step.isKeyStep && <span className="key-indicator">★</span>}
                </div>
                <div className="step-title">
                  <h3>{step.description}</h3>
                </div>
                <div className="step-toggle">
                  <span>{isExpanded ? '▼' : '►'}</span>
                </div>
              </div>
              
              {isExpanded && (
                <div 
                  className="step-content"
                  id={`step-content-${stepId}`}
                >
                  {/* Step latex expression */}
                  {step.latex && (
                    <div className="step-expression">
                      <LatexRenderer 
                        latex={step.latex}
                        isBlock={true}
                      />
                    </div>
                  )}
                  
                  {/* Step explanation */}
                  {step.explanation && (
                    <div className="step-explanation">
                      <p>{step.explanation}</p>
                    </div>
                  )}
                  
                  {/* Step hints or notes if available */}
                  {step.hints && step.hints.length > 0 && (
                    <div className="step-hints">
                      <h4>Hints:</h4>
                      <ul>
                        {step.hints.map((hint, index) => (
                          <li key={index}>{hint}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default StepByStepSolution;

import React, { useState, useEffect } from 'react';
import StepByStepSolution from '../math/StepByStepSolution';
import VisualizationDisplay from '../visualization/VisualizationDisplay';
import LatexRenderer from '../latex/LatexRenderer';
import preferencesService from '../../services/preferencesService';
import './IntegratedResponse.css';

/**
 * Integrated Response Component
 * A comprehensive display of mathematical results
 * 
 * @param {Object} props - Component props
 * @param {Object} props.response - The response data to display
 * @param {string} props.response.text - Text response
 * @param {Array} props.response.steps - Solution steps
 * @param {Array} props.response.visualizations - Visualizations
 * @param {Array} props.response.latex_expressions - LaTeX expressions
 * @param {string} props.className - Additional CSS class
 * @param {boolean} props.loading - Whether the response is loading
 * @param {string} props.error - Error message if any
 */
const IntegratedResponse = ({
  response = null,
  className = '',
  loading = false,
  error = ''
}) => {
  // State for active tab
  const [activeTab, setActiveTab] = useState('full');
  
  // Check if we have content for each tab
  const hasSteps = response && response.steps && response.steps.length > 0;
  const hasVisualizations = response && response.visualizations && response.visualizations.length > 0;
  const hasLatex = response && response.latex_expressions && response.latex_expressions.length > 0;
  
  // Get response preferences
  const responsePrefs = preferencesService.get('response');
  
  // Set the active tab based on content and preferences
  useEffect(() => {
    if (response) {
      if (activeTab === 'steps' && !hasSteps) {
        setActiveTab('full');
      } else if (activeTab === 'visualizations' && !hasVisualizations) {
        setActiveTab('full');
      } else if (activeTab === 'latex' && !hasLatex) {
        setActiveTab('full');
      }
    }
  }, [response, activeTab, hasSteps, hasVisualizations, hasLatex]);
  
  // Handle tab change
  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };
  
  // If there's no response and we're not loading, return nothing
  if (!response && !loading && !error) {
    return null;
  }
  
  return (
    <div className={`integrated-response ${className}`}>
      {/* Loading state */}
      {loading && (
        <div className="response-loading" aria-live="polite">
          <div className="loading-indicator"></div>
          <p>Processing your request...</p>
        </div>
      )}
      
      {/* Error state */}
      {error && (
        <div className="response-error" aria-live="assertive">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}
      
      {/* Response content */}
      {response && !loading && !error && (
        <>
          {/* Tab navigation */}
          <div className="response-tabs">
            <button 
              className={`tab-button ${activeTab === 'full' ? 'active' : ''}`}
              onClick={() => handleTabChange('full')}
              aria-label="Full response"
              aria-selected={activeTab === 'full'}
              role="tab"
            >
              Full Response
            </button>
            {hasSteps && (
              <button 
                className={`tab-button ${activeTab === 'steps' ? 'active' : ''}`}
                onClick={() => handleTabChange('steps')}
                aria-label="Step-by-step solution"
                aria-selected={activeTab === 'steps'}
                role="tab"
              >
                Step-by-Step
              </button>
            )}
            {hasVisualizations && (
              <button 
                className={`tab-button ${activeTab === 'visualizations' ? 'active' : ''}`}
                onClick={() => handleTabChange('visualizations')}
                aria-label="Visualizations"
                aria-selected={activeTab === 'visualizations'}
                role="tab"
              >
                Visualizations
              </button>
            )}
            {hasLatex && (
              <button 
                className={`tab-button ${activeTab === 'latex' ? 'active' : ''}`}
                onClick={() => handleTabChange('latex')}
                aria-label="LaTeX code"
                aria-selected={activeTab === 'latex'}
                role="tab"
              >
                LaTeX
              </button>
            )}
          </div>
          
          {/* Tab content */}
          <div className="response-content" role="tabpanel" aria-label={`${activeTab} content`}>
            {/* Full response tab */}
            {activeTab === 'full' && (
              <div className="full-response">
                <div className="response-text">
                  {/* Parse response text to render LaTeX */}
                  {response.text.split(/(\$\$[\s\S]*?\$\$|\$[\s\S]*?\$)/).map((part, index) => {
                    // Check if this is a LaTeX block
                    const isDisplayMath = part.startsWith('$$') && part.endsWith('$$');
                    const isInlineMath = part.startsWith('$') && part.endsWith('$') && !isDisplayMath;
                    
                    if (isDisplayMath) {
                      // Extract LaTeX without delimiters
                      const latex = part.slice(2, -2);
                      return (
                        <div key={index} className="display-math">
                          <LatexRenderer latex={latex} isBlock={true} />
                        </div>
                      );
                    } else if (isInlineMath) {
                      // Extract LaTeX without delimiters
                      const latex = part.slice(1, -1);
                      return (
                        <span key={index} className="inline-math">
                          <LatexRenderer latex={latex} isBlock={false} />
                        </span>
                      );
                    } else {
                      // Regular text
                      return <span key={index}>{part}</span>;
                    }
                  })}
                </div>
                
                {/* Include step-by-step in full response if preference is set */}
                {hasSteps && responsePrefs.showStepByStep && (
                  <div className="response-steps">
                    <h3>Solution Steps</h3>
                    <StepByStepSolution 
                      steps={response.steps}
                      detailLevel={responsePrefs.detailLevel}
                    />
                  </div>
                )}
                
                {/* Include visualizations in full response if preference is set */}
                {hasVisualizations && responsePrefs.showVisualizations && (
                  <div className="response-visualizations">
                    <h3>Visualizations</h3>
                    {response.visualizations.map((viz, index) => (
                      <div key={index} className="response-visualization-item">
                        <VisualizationDisplay 
                          visualization={viz}
                          type={viz.type}
                          data={viz.data}
                          title={viz.title}
                          description={viz.description}
                        />
                      </div>
                    ))}
                  </div>
                )}
                
                {/* Key concepts section if available */}
                {response.raw && response.raw.key_concepts && (
                  <div className="key-concepts">
                    <h3>Key Concepts</h3>
                    <ul>
                      {response.raw.key_concepts.map((concept, index) => (
                        <li key={index}>{concept}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* References section if available */}
                {response.raw && response.raw.references && (
                  <div className="references">
                    <h3>References</h3>
                    <ul>
                      {response.raw.references.map((ref, index) => (
                        <li key={index}>{ref}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
            
            {/* Step-by-step tab */}
            {activeTab === 'steps' && hasSteps && (
              <div className="steps-tab">
                <StepByStepSolution 
                  steps={response.steps}
                  detailLevel={responsePrefs.detailLevel}
                  expandAll={true}
                />
              </div>
            )}
            
            {/* Visualizations tab */}
            {activeTab === 'visualizations' && hasVisualizations && (
              <div className="visualizations-tab">
                {response.visualizations.map((viz, index) => (
                  <div key={index} className="visualization-item">
                    <h3>{viz.title}</h3>
                    {viz.description && <p>{viz.description}</p>}
                    <VisualizationDisplay 
                      visualization={viz}
                      type={viz.type}
                      data={viz.data}
                      fullSize={true}
                    />
                  </div>
                ))}
              </div>
            )}
            
            {/* LaTeX tab */}
            {activeTab === 'latex' && hasLatex && (
              <div className="latex-tab">
                <h3>LaTeX Expressions</h3>
                {response.latex_expressions.map((latex, index) => (
                  <div key={index} className="latex-item">
                    <div className="latex-display">
                      <LatexRenderer latex={latex} isBlock={true} />
                    </div>
                    <div className="latex-code">
                      <pre>{latex}</pre>
                      <button 
                        className="copy-button"
                        onClick={() => navigator.clipboard.writeText(latex)}
                        aria-label="Copy LaTeX code"
                      >
                        Copy
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {/* Export options */}
          <div className="response-actions">
            <button 
              className="action-button"
              onClick={() => {
                // Create a formatted version of the response
                const formattedResponse = `
                  # Mathematical Response
                  
                  ${response.text}
                  
                  ${hasSteps ? '## Solution Steps\n\n' + response.steps.map(step => 
                    `${step.number}. ${step.description}\n${step.explanation}`
                  ).join('\n\n') : ''}
                  
                  ${response.raw && response.raw.key_concepts ? '## Key Concepts\n\n' + 
                    response.raw.key_concepts.map(concept => `- ${concept}`).join('\n') : ''}
                  
                  ${response.raw && response.raw.references ? '## References\n\n' + 
                    response.raw.references.map(ref => `- ${ref}`).join('\n') : ''}
                `;
                
                // Create a blob and download link
                const blob = new Blob([formattedResponse], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'mathematical_response.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
              aria-label="Export response"
            >
              Export
            </button>
            
            <button 
              className="action-button"
              onClick={() => window.print()}
              aria-label="Print response"
            >
              Print
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default IntegratedResponse;

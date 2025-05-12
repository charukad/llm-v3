import React, { useState, useEffect } from 'react';
import VisualizationDisplay from './VisualizationDisplay';
import './InteractionVisualizations.css';

/**
 * Component for displaying visualizations related to an interaction
 */
const InteractionVisualizations = ({ interactionId }) => {
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedVisualization, setSelectedVisualization] = useState(null);
  
  useEffect(() => {
    if (!interactionId) {
      setLoading(false);
      return;
    }
    
    const fetchVisualizations = async () => {
      try {
        const response = await fetch(`/api/visualization/by-interaction/${interactionId}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch visualizations: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.visualizations && data.visualizations.length > 0) {
          setVisualizations(data.visualizations);
          setSelectedVisualization(data.visualizations[0]);
        } else {
          setVisualizations([]);
          setSelectedVisualization(null);
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching visualizations:', err);
        setError(`Failed to fetch visualizations: ${err.message}`);
        setLoading(false);
      }
    };
    
    fetchVisualizations();
  }, [interactionId]);
  
  if (loading) {
    return <div className="interaction-visualizations-loading">Loading visualizations...</div>;
  }
  
  if (error) {
    return <div className="interaction-visualizations-error">Error: {error}</div>;
  }
  
  if (visualizations.length === 0) {
    return <div className="interaction-visualizations-empty">No visualizations available for this interaction</div>;
  }
  
  return (
    <div className="interaction-visualizations">
      <h3>Visualizations</h3>
      
      <div className="visualization-tabs">
        {visualizations.map((viz) => (
          <button
            key={viz._id}
            className={`visualization-tab ${selectedVisualization && selectedVisualization._id === viz._id ? 'active' : ''}`}
            onClick={() => setSelectedVisualization(viz)}
          >
            {viz.visualization_type.replace(/_/g, ' ').split(' ').map(word => 
              word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ')}
          </button>
        ))}
      </div>
      
      <div className="selected-visualization">
        {selectedVisualization && (
          <VisualizationDisplay 
            visualizationId={selectedVisualization._id} 
            visualizationData={selectedVisualization}
          />
        )}
      </div>
    </div>
  );
};

export default InteractionVisualizations;

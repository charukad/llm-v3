import React, { useState, useEffect } from 'react';
import './VisualizationSelector.css';

/**
 * Component for selecting and requesting visualizations
 */
const VisualizationSelector = ({ 
  mathematicalContext,
  onVisualizationGenerated,
  interactionId
}) => {
  const [availableTypes, setAvailableTypes] = useState([]);
  const [selectedType, setSelectedType] = useState('');
  const [parameters, setParameters] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  
  // Fetch available visualization types
  useEffect(() => {
    const fetchVisualizationTypes = async () => {
      try {
        const response = await fetch('/api/visualization/types');
        
        if (!response.ok) {
          throw new Error(`Failed to fetch visualization types: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Combine all visualization types
        const allTypes = [
          ...(data.base_types || []),
          ...(data.advanced_types || [])
        ];
        
        // Remove duplicates
        const uniqueTypes = [...new Set(allTypes)];
        
        setAvailableTypes(uniqueTypes);
      } catch (err) {
        console.error('Error fetching visualization types:', err);
        setError(`Failed to fetch visualization types: ${err.message}`);
      }
    };
    
    fetchVisualizationTypes();
  }, []);
  
  // Get recommendations when mathematical context changes
  useEffect(() => {
    if (!mathematicalContext) return;
    
    const getRecommendations = async () => {
      try {
        const response = await fetch('/api/visualization/select', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(mathematicalContext)
        });
        
        if (!response.ok) {
          throw new Error(`Failed to get recommendations: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success && data.recommended_visualization) {
          setRecommendations([
            data.recommended_visualization,
            ...(data.alternative_visualizations || [])
          ]);
          
          // Auto-select the recommended visualization type
          setSelectedType(data.recommended_visualization.type);
          
          // Set the recommended parameters
          setParameters(data.recommended_visualization.params || {});
        }
      } catch (err) {
        console.error('Error getting visualization recommendations:', err);
      }
    };
    
    getRecommendations();
  }, [mathematicalContext]);
  
  // Handle type selection
  const handleTypeChange = (e) => {
    const newType = e.target.value;
    setSelectedType(newType);
    
    // Find matching recommendation
    const recommendation = recommendations.find(rec => rec.type === newType);
    if (recommendation && recommendation.params) {
      setParameters(recommendation.params);
    } else {
      // Reset parameters if no matching recommendation
      setParameters({});
    }
  };
  
  // Handle parameter change
  const handleParameterChange = (key, value) => {
    setParameters(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  // Generate visualization
  const handleGenerateVisualization = async () => {
    if (!selectedType) {
      setError('Please select a visualization type');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('visualization_type', selectedType);
      formData.append('params', JSON.stringify(parameters));
      
      if (interactionId) {
        formData.append('interaction_id', interactionId);
      }
      
      // Send request
      const response = await fetch('/api/visualization/generate', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Failed to generate visualization: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Notify parent component
        if (onVisualizationGenerated) {
          onVisualizationGenerated(result);
        }
      } else {
        setError(result.error || 'Failed to generate visualization');
      }
    } catch (err) {
      console.error('Error generating visualization:', err);
      setError(`Failed to generate visualization: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Render parameter input based on parameter type
  const renderParameterInput = (key, value) => {
    // Determine input type based on value type
    if (typeof value === 'boolean') {
      return (
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => handleParameterChange(key, e.target.checked)}
        />
      );
    }
    
    if (typeof value === 'number') {
      return (
        <input
          type="number"
          value={value}
          onChange={(e) => handleParameterChange(key, parseFloat(e.target.value))}
        />
      );
    }
    
    if (Array.isArray(value)) {
      // For arrays, create a simple text input with comma-separated values
      return (
        <input
          type="text"
          value={value.join(', ')}
          onChange={(e) => {
            const newValue = e.target.value.split(',').map(item => {
              const trimmed = item.trim();
              return isNaN(parseFloat(trimmed)) ? trimmed : parseFloat(trimmed);
            });
            handleParameterChange(key, newValue);
          }}
        />
      );
    }
    
    // Default to text input
    return (
      <input
        type="text"
        value={value}
        onChange={(e) => handleParameterChange(key, e.target.value)}
      />
    );
  };
  
  return (
    <div className="visualization-selector">
      <h3>Generate Visualization</h3>
      
      <div className="visualization-type-selector">
        <label htmlFor="visualization-type">Visualization Type:</label>
        <select
          id="visualization-type"
          value={selectedType}
          onChange={handleTypeChange}
        >
          <option value="">Select a visualization type</option>
          {availableTypes.map(type => (
            <option key={type} value={type}>
              {type.replace(/_/g, ' ').split(' ').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
              ).join(' ')}
            </option>
          ))}
        </select>
      </div>
      
      {selectedType && (
        <div className="visualization-parameters">
          <h4>Parameters</h4>
          {Object.entries(parameters).map(([key, value]) => (
            <div key={key} className="parameter-row">
              <label htmlFor={`param-${key}`}>
                {key.replace(/_/g, ' ').split(' ').map(word => 
                  word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' ')}:
              </label>
              {renderParameterInput(key, value)}
            </div>
          ))}
        </div>
      )}
      
      {error && (
        <div className="visualization-error">
          {error}
        </div>
      )}
      
      <div className="visualization-actions">
        <button
          onClick={handleGenerateVisualization}
          disabled={loading || !selectedType}
        >
          {loading ? 'Generating...' : 'Generate Visualization'}
        </button>
      </div>
    </div>
  );
};

export default VisualizationSelector;

import React, { useEffect, useRef, useState } from 'react';
import './VisualizationDisplay.css';

/**
 * VisualizationDisplay Component
 * Renders different types of mathematical visualizations
 * 
 * @param {Object} props - Component props
 * @param {Object} props.visualization - The visualization object
 * @param {string} props.type - Visualization type ('image', 'svg', 'plotly', 'canvas')
 * @param {Object} props.data - Visualization data
 * @param {string} props.title - Visualization title
 * @param {string} props.description - Visualization description
 * @param {boolean} props.fullSize - Whether to display the visualization at full size
 * @param {string} props.className - Additional CSS class
 */
const VisualizationDisplay = ({
  visualization = null,
  type = 'image',
  data = null,
  title = '',
  description = '',
  fullSize = false,
  className = ''
}) => {
  // State for tracking loading and error states
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [visualizationData, setVisualizationData] = useState(null);
  
  // Refs for DOM elements
  const containerRef = useRef(null);
  const plotlyRef = useRef(null);
  const canvasRef = useRef(null);
  
  // Use either provided props or visualization object properties
  const visualizationType = type || (visualization && visualization.type) || 'image';
  const visualizationTitle = title || (visualization && visualization.title) || '';
  const visualizationDescription = description || (visualization && visualization.description) || '';
  const derivedVisualizationData = data || (visualization && visualization.data) || null;
  
  // URL for image or SVG type visualizations
  const visualizationUrl = visualization && visualization.url ? visualization.url : '';
  
  // Effect to process and display the visualization based on type
  useEffect(() => {
    setLoading(true);
    setError('');
    
    const processVisualization = async () => {
      try {
        switch (visualizationType) {
          case 'image':
            // For image type, just set loading to false since the image will load via img tag
            setLoading(false);
            break;
            
          case 'svg':
            // For SVG type, we might need to fetch the SVG content if it's not provided directly
            if (visualizationUrl && !derivedVisualizationData) {
              const response = await fetch(visualizationUrl);
              if (!response.ok) {
                throw new Error(`Failed to load SVG: ${response.statusText}`);
              }
              const svgText = await response.text();
              setVisualizationData(svgText);
            } else if (derivedVisualizationData) {
              // If SVG data is provided directly, use it
              setVisualizationData(derivedVisualizationData);
            } else {
              throw new Error('No SVG data or URL provided');
            }
            setLoading(false);
            break;
            
          case 'plotly':
            // For Plotly visualizations, we need to render using the Plotly library
            if (!window.Plotly) {
              throw new Error('Plotly library not loaded');
            }
            
            if (!derivedVisualizationData) {
              throw new Error('No Plotly data provided');
            }
            
            // Clear any existing plot
            if (plotlyRef.current) {
              window.Plotly.purge(plotlyRef.current);
            }
            
            // Render the new plot
            window.Plotly.newPlot(
              plotlyRef.current,
              derivedVisualizationData.data || [],
              derivedVisualizationData.layout || {},
              derivedVisualizationData.config || { responsive: true }
            );
            
            setLoading(false);
            break;
            
          case 'canvas':
            // For canvas visualizations, we need to draw on the canvas element
            if (!derivedVisualizationData || !derivedVisualizationData.drawFunction) {
              throw new Error('No canvas drawing function provided');
            }
            
            const canvas = canvasRef.current;
            if (!canvas) {
              throw new Error('Canvas element not available');
            }
            
            const ctx = canvas.getContext('2d');
            
            // Clear the canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Execute the drawing function
            derivedVisualizationData.drawFunction(ctx, canvas.width, canvas.height);
            
            setLoading(false);
            break;
            
          default:
            throw new Error(`Unsupported visualization type: ${visualizationType}`);
        }
      } catch (error) {
        console.error('Visualization error:', error);
        setError(error.message || 'Failed to load visualization');
        setLoading(false);
      }
    };
    
    processVisualization();
    
    // Cleanup function
    return () => {
      if (visualizationType === 'plotly' && plotlyRef.current && window.Plotly) {
        window.Plotly.purge(plotlyRef.current);
      }
    };
  }, [visualizationType, visualizationUrl, derivedVisualizationData]);
  
  // Handle resize events for responsive visualizations
  useEffect(() => {
    const handleResize = () => {
      if (visualizationType === 'plotly' && plotlyRef.current && window.Plotly) {
        window.Plotly.Plots.resize(plotlyRef.current);
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [visualizationType]);
  
  // Calculate CSS classes
  const containerClasses = `visualization-display ${visualizationType} ${fullSize ? 'full-size' : ''} ${className}`;
  
  return (
    <div 
      ref={containerRef} 
      className={containerClasses}
      aria-label={visualizationTitle}
      aria-describedby={`viz-description-${title ? title.replace(/\s+/g, '-').toLowerCase() : 'untitled'}`}
    >
      {/* Loading state */}
      {loading && (
        <div className="visualization-loading" aria-live="polite">
          <div className="loading-indicator"></div>
          <p>Loading visualization...</p>
        </div>
      )}
      
      {/* Error state */}
      {error && (
        <div className="visualization-error" aria-live="assertive">
          <p>Error: {error}</p>
        </div>
      )}
      
      {/* Image visualization */}
      {!loading && !error && visualizationType === 'image' && visualizationUrl && (
        <div className="visualization-image-container">
          <img 
            src={visualizationUrl} 
            alt={visualizationDescription || visualizationTitle || 'Mathematical visualization'} 
            className="visualization-image"
            onLoad={() => setLoading(false)}
            onError={() => setError('Failed to load image')}
          />
        </div>
      )}
      
      {/* SVG visualization */}
      {!loading && !error && visualizationType === 'svg' && (
        <div 
          className="visualization-svg-container"
          dangerouslySetInnerHTML={{ __html: visualizationData }}
        />
      )}
      
      {/* Plotly visualization */}
      {!loading && !error && visualizationType === 'plotly' && (
        <div 
          ref={plotlyRef}
          className="visualization-plotly-container"
        />
      )}
      
      {/* Canvas visualization */}
      {!loading && !error && visualizationType === 'canvas' && (
        <canvas 
          ref={canvasRef}
          className="visualization-canvas"
          width={fullSize ? 800 : 400}
          height={fullSize ? 600 : 300}
        />
      )}
      
      {/* Visualization description */}
      {!loading && !error && visualizationDescription && (
        <div 
          id={`viz-description-${title ? title.replace(/\s+/g, '-').toLowerCase() : 'untitled'}`}
          className="visualization-description"
        >
          <p>{visualizationDescription}</p>
        </div>
      )}
      
      {/* Controls for interactive visualizations */}
      {!loading && !error && (visualizationType === 'plotly' || visualizationType === 'canvas') && (
        <div className="visualization-controls">
          {/* Zoom controls for canvas */}
          {visualizationType === 'canvas' && (
            <>
              <button 
                className="control-button"
                onClick={() => {
                  if (derivedVisualizationData && derivedVisualizationData.zoomIn && canvasRef.current) {
                    derivedVisualizationData.zoomIn(canvasRef.current.getContext('2d'));
                  }
                }}
                aria-label="Zoom in"
              >
                Zoom In
              </button>
              <button 
                className="control-button"
                onClick={() => {
                  if (derivedVisualizationData && derivedVisualizationData.zoomOut && canvasRef.current) {
                    derivedVisualizationData.zoomOut(canvasRef.current.getContext('2d'));
                  }
                }}
                aria-label="Zoom out"
              >
                Zoom Out
              </button>
              <button 
                className="control-button"
                onClick={() => {
                  if (derivedVisualizationData && derivedVisualizationData.reset && canvasRef.current) {
                    derivedVisualizationData.reset(canvasRef.current.getContext('2d'));
                  }
                }}
                aria-label="Reset view"
              >
                Reset
              </button>
            </>
          )}
          
          {/* Plotly already has its own controls */}
        </div>
      )}
    </div>
  );
};

export default VisualizationDisplay;

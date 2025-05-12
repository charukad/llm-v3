/**
 * Service for handling mathematical responses in the Mathematical Multimodal LLM system
 * Processes and formats responses from the backend API
 */

import apiService from './apiService';

class ResponseService {
  // Format a mathematical response for display
  formatResponse(response) {
    if (!response || !response.data) {
      return {
        text: 'No response data available',
        steps: [],
        visualizations: [],
        latex_expressions: []
      };
    }
    
    // Extract and format step-by-step solution
    const steps = this._formatSteps(response.data.steps || []);
    
    // Extract and format visualizations
    const visualizations = this._formatVisualizations(response.data.visualizations || []);
    
    // Extract LaTeX expressions
    const latexExpressions = response.data.latex_expressions || [];
    
    return {
      text: response.data.text || '',
      steps,
      visualizations,
      latex_expressions: latexExpressions,
      raw: response.data
    };
  }
  
  // Format step-by-step solution
  _formatSteps(steps) {
    return steps.map((step, index) => ({
      id: step.id || `step-${index}`,
      number: step.number || index + 1,
      type: step.type || 'general',
      description: step.description || '',
      latex: step.latex || '',
      explanation: step.explanation || '',
      isKeyStep: step.is_key_step || false
    }));
  }
  
  // Format visualizations
  _formatVisualizations(visualizations) {
    return visualizations.map((viz, index) => ({
      id: viz.id || `viz-${index}`,
      type: viz.type || 'image',
      title: viz.title || `Visualization ${index + 1}`,
      description: viz.description || '',
      url: viz.url || '',
      data: viz.data || null,
      parameters: viz.parameters || {}
    }));
  }
  
  // Get a full mathematical response by ID
  async getResponseById(responseId) {
    try {
      const response = await apiService.workflows.getWorkflowResult(responseId);
      if (response.success) {
        return {
          success: true,
          ...this.formatResponse(response)
        };
      }
      return response;
    } catch (error) {
      return {
        success: false,
        error: error.message || 'Failed to get response'
      };
    }
  }
  
  // Get step-by-step solution by query ID
  async getStepByStepSolution(queryId) {
    try {
      const response = await apiService.math.getStepByStepSolution(queryId);
      if (response.success) {
        return {
          success: true,
          steps: this._formatSteps(response.data.steps || [])
        };
      }
      return response;
    } catch (error) {
      return {
        success: false,
        error: error.message || 'Failed to get step-by-step solution'
      };
    }
  }
  
  // Convert natural language to LaTeX
  async convertTextToLatex(text) {
    return apiService.math.nlToLatex(text);
  }
  
  // Parse and validate LaTeX expression
  async validateLatex(latexExpression) {
    return apiService.math.parseLatex(latexExpression);
  }
}

// Create a singleton instance
const responseService = new ResponseService();
export default responseService;

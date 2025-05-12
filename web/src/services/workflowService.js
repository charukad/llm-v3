/**
 * Service for handling workflow operations in the Mathematical Multimodal LLM system
 * Communicates with the backend API to manage mathematical workflows
 */

import apiService from "./apiService";

// Enable debugging mode
const DEBUG = true;

class WorkflowService {
  // Debug logger
  _debug(...args) {
    if (DEBUG) {
      console.log("[WorkflowService]", ...args);
    }
  }

  // Start a new math problem workflow
  async startMathProblemWorkflow(query, options = {}) {
    this._debug("Starting math problem workflow with query:", query);

    try {
      const response = await apiService.math.submitQuery({
        query,
        options,
      });

      this._debug("Math query response:", response);

      if (response.success) {
        // Return the workflow ID for status tracking
        return {
          success: true,
          workflowId: response.data.workflow_id,
          message: "Math problem workflow started successfully",
        };
      }

      return response;
    } catch (error) {
      this._debug("Error in startMathProblemWorkflow:", error);
      return {
        success: false,
        error: error.message || "Failed to start math problem workflow",
      };
    }
  }

  // Start a handwriting recognition workflow
  async startHandwritingWorkflow(imageData, options = {}) {
    this._debug("Starting handwriting workflow");

    try {
      const response = await apiService.multimodal.processHandwriting(
        imageData
      );

      this._debug("Handwriting response:", response);

      if (response.success) {
        return {
          success: true,
          workflowId: response.data.workflow_id,
          recognizedLatex: response.data.recognized_latex,
          confidence: response.data.confidence,
          message: "Handwriting recognition workflow started successfully",
        };
      }

      return response;
    } catch (error) {
      this._debug("Error in startHandwritingWorkflow:", error);
      return {
        success: false,
        error: error.message || "Failed to start handwriting workflow",
      };
    }
  }

  // Poll for workflow status until completion or timeout
  async pollWorkflowStatus(workflowId, maxAttempts = 30, intervalMs = 1000) {
    this._debug("Polling workflow status for ID:", workflowId);
    let attempts = 0;

    // Use mock data for testing if the backend workflow API isn't available
    const useMockData = apiService.USE_MOCK && apiService.USE_MOCK.WORKFLOW;
    if (useMockData) {
      this._debug("Using mock workflow data");

      // Simulate a short delay
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Get mock result from session storage or create one
      let mockResult = null;
      const storedQuery = sessionStorage.getItem(workflowId);

      if (storedQuery) {
        this._debug("Found stored query for workflow:", workflowId);
        const queryData = JSON.parse(storedQuery);
        mockResult = {
          success: true,
          data:
            apiService.MOCK_DATA.mathQueries[queryData.query] ||
            apiService.MOCK_DATA.defaultMathResponse,
        };
      } else {
        this._debug("No stored query found, using default mock response");
        mockResult = {
          success: true,
          data: apiService.MOCK_DATA.defaultMathResponse,
        };
      }

      return mockResult;
    }

    // Create a promise that resolves when the workflow completes
    return new Promise((resolve, reject) => {
      const checkStatus = async () => {
        attempts++;
        this._debug(`Checking status attempt ${attempts}/${maxAttempts}`);

        try {
          const response = await apiService.workflows.getWorkflowStatus(
            workflowId
          );
          this._debug("Status response:", response);

          if (!response.success) {
            this._debug("Status check failed:", response.error);
            clearInterval(intervalId);
            reject(response.error);
            return;
          }

          const status = response.data.status;
          this._debug("Workflow status:", status);

          // Check if workflow completed or errored
          if (status === "completed") {
            this._debug("Workflow completed, getting results");
            clearInterval(intervalId);

            // Get the result
            const resultResponse = await apiService.workflows.getWorkflowResult(
              workflowId
            );
            this._debug("Result response:", resultResponse);
            resolve(resultResponse);
            return;
          } else if (status === "error") {
            this._debug("Workflow error:", response.data.error);
            clearInterval(intervalId);
            reject(new Error(response.data.error || "Workflow failed"));
            return;
          }

          // Check if we've exceeded max attempts
          if (attempts >= maxAttempts) {
            this._debug(
              "Workflow polling timed out after",
              maxAttempts,
              "attempts"
            );
            clearInterval(intervalId);
            reject(new Error("Workflow polling timed out"));
            return;
          }
        } catch (error) {
          this._debug("Error in checkStatus:", error);
          clearInterval(intervalId);
          reject(error);
        }
      };

      // Start polling at the specified interval
      const intervalId = setInterval(checkStatus, intervalMs);

      // Perform an initial check immediately
      checkStatus();
    });
  }

  // Start a workflow and wait for its completion
  async executeWorkflow(workflowType, data, options = {}) {
    this._debug(
      "Executing workflow of type:",
      workflowType,
      "with data:",
      data
    );
    let workflowId;

    try {
      // Start the appropriate workflow type
      if (workflowType === "math") {
        const response = await this.startMathProblemWorkflow(data, options);
        this._debug("Math workflow started with response:", response);
        if (!response.success) return response;
        workflowId = response.workflowId;
      } else if (workflowType === "handwriting") {
        const response = await this.startHandwritingWorkflow(data, options);
        this._debug("Handwriting workflow started with response:", response);
        if (!response.success) return response;
        workflowId = response.workflowId;
      } else {
        return {
          success: false,
          error: `Unknown workflow type: ${workflowType}`,
        };
      }

      // Poll for the workflow completion
      this._debug("Starting to poll for workflow completion");
      const result = await this.pollWorkflowStatus(workflowId);
      this._debug("Workflow completed with result:", result);
      return result;
    } catch (error) {
      this._debug("Error in executeWorkflow:", error);
      return {
        success: false,
        error: error.message || "Failed to execute workflow",
        workflowId,
      };
    }
  }

  // Get visualization from a workflow
  async getWorkflowVisualization(workflowId, visualizationId) {
    this._debug(
      "Getting visualization",
      visualizationId,
      "from workflow",
      workflowId
    );
    return apiService.visualization.getVisualization(visualizationId);
  }
}

// Create a singleton instance
const workflowService = new WorkflowService();
export default workflowService;

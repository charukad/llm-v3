class WorkflowService {
  constructor() {
    this.baseUrl = "/api/workflow";
    this.maxPollingAttempts = 30; // 30 seconds max
    this.pollingInterval = 1000; // 1 second
  }

  async pollWorkflowStatus(
    workflowId,
    maxAttempts = this.maxPollingAttempts,
    intervalMs = this.pollingInterval
  ) {
    console.log(
      `[WorkflowService] Starting to poll workflow status for ID: ${workflowId}`
    );

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        console.log(
          `[WorkflowService] Polling attempt ${attempt}/${maxAttempts}`
        );

        const response = await fetch(`${this.baseUrl}/status/${workflowId}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log(`[WorkflowService] Workflow status:`, data);

        if (data.status === "completed") {
          console.log(`[WorkflowService] Workflow completed successfully`);
          return data;
        } else if (data.status === "failed") {
          console.error(`[WorkflowService] Workflow failed:`, data.error);
          throw new Error(data.error || "Workflow failed");
        }

        // Wait before next attempt
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      } catch (error) {
        console.error(
          `[WorkflowService] Error polling workflow status:`,
          error
        );
        throw error;
      }
    }

    throw new Error("Workflow polling timed out");
  }

  async executeWorkflow(workflowType, data, options = {}) {
    console.log(`[WorkflowService] Starting workflow execution:`, {
      workflowType,
      data,
    });

    try {
      // Start workflow
      const response = await fetch(`${this.baseUrl}/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          workflow_type: workflowType,
          data: data,
          options: options,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const { workflow_id } = await response.json();
      console.log(`[WorkflowService] Workflow started with ID: ${workflow_id}`);

      // Poll for completion
      return await this.pollWorkflowStatus(workflow_id);
    } catch (error) {
      console.error(`[WorkflowService] Error executing workflow:`, error);
      throw error;
    }
  }
}

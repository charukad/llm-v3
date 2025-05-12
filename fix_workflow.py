@router.get("/{workflow_id}/result")
async def get_workflow_result(workflow_id: str):
    """
    Get the result of a completed workflow.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        Workflow result details
    """
    # Check if workflow exists
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    workflow = workflows[workflow_id]
    
    # Check if workflow is completed
    if workflow["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Workflow {workflow_id} is not completed yet. Current status: {workflow['status']}"
        )
    
    # For demo purposes, generate a result based on a default query or the recorded query
    query = workflow.get("query") or "Calculate the integral of ln(x) from 1 to 6"
    
    # Try to find a matching result in our samples
    if query in SAMPLE_RESULTS:
        result = SAMPLE_RESULTS[query]
    else:
        # Default result
        result = {
            "steps": [
                {
                    "description": "This is a generated response for demonstration",
                    "latex": "\\text{Result for: } " + query,
                }
            ],
            "answer": f"Result for: {query}",
            "text": f"This is a generated response for the query: {query}",
            "visualizations": []
        }
    
    # Store the result in the workflow
    workflow["result"] = result
    
    return result 
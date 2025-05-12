"""Tests for prompt routing functionality."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from api.rest.models import InputRequest
from api.rest.routes.multimodal import process_input, orchestration_manager

class TestPromptRouting:
    """Test suite for prompt routing functionality."""

    @pytest_asyncio.fixture
    async def setup(self):
        """Setup test environment."""
        # Mock the workflow registry
        with patch('orchestration.manager.orchestration_manager.OrchestrationManager.start_workflow') as mock_start_workflow:
            mock_start_workflow.return_value = "test_workflow_id"
            yield mock_start_workflow

    @pytest.mark.asyncio
    async def test_text_prompt_routing(self, setup):
        """Test routing of simple text prompts."""
        # Test prompt
        input_request = InputRequest(
            content="What is the capital of France?",
            input_type="text",
            conversation_id="test_conv_1"
        )

        # Process input
        result = await process_input(input_request)

        # Add assertions here
        assert result is not None
        assert "routing" in result
        assert result["routing"]["agent_type"] == "core_llm"
        assert result["workflow_id"] == "test_workflow_id"

    @pytest.mark.asyncio
    async def test_math_prompt_routing(self, setup):
        """Test routing of mathematical prompts."""
        # Test prompt
        input_request = InputRequest(
            content="Solve the equation: 2x + 3 = 7",
            input_type="text",
            conversation_id="test_conv_2"
        )

        # Process input
        result = await process_input(input_request)

        # Add assertions here
        assert result is not None
        assert "routing" in result
        assert result["routing"]["agent_type"] == "core_llm"
        assert result["workflow_id"] == "test_workflow_id"

    @pytest.mark.asyncio
    async def test_mixed_prompt_routing(self, setup):
        """Test routing of prompts containing both text and math."""
        # Test prompt
        input_request = InputRequest(
            content="Can you explain how to solve the quadratic equation x^2 + 5x + 6 = 0?",
            input_type="text",
            conversation_id="test_conv_3"
        )

        # Process input
        result = await process_input(input_request)

        # Add assertions here
        assert result is not None
        assert "routing" in result
        assert result["routing"]["agent_type"] == "core_llm"
        assert result["workflow_id"] == "test_workflow_id"

    @pytest.mark.asyncio
    async def test_error_handling(self, setup):
        """Test error handling for invalid prompts."""
        # Test with empty content
        input_request = InputRequest(
            content="",
            input_type="text",
            conversation_id="test_conv_4"
        )

        # Process input
        with pytest.raises(HTTPException) as exc_info:
            await process_input(input_request)

        assert exc_info.value.status_code == 400
        assert "Empty content" in str(exc_info.value.detail)

if __name__ == "__main__":
    pytest.main(["-v"]) 
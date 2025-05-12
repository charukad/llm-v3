"""
Comprehensive API integration tests.
Tests all major API endpoints and their interactions with the system.
"""
import pytest
import os
import requests
import json
import time
import tempfile
import base64
from PIL import Image, ImageDraw, ImageFont
import io

class TestAPIIntegration:
    """Test class for comprehensive API integration tests."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Base URL for API
        cls.base_url = "http://localhost:8000/api"
        
        # Create test user for auth if needed
        cls.test_user = {
            "username": "apitest_user",
            "password": "testpassword123"
        }
        
        # Create authentication token if needed
        # This is just a placeholder - adjust based on your actual auth system
        auth_response = requests.post(
            f"{cls.base_url}/auth/login",
            json=cls.test_user
        )
        
        if auth_response.status_code == 200:
            cls.auth_token = auth_response.json().get("token")
            cls.headers = {"Authorization": f"Bearer {cls.auth_token}"}
        else:
            cls.headers = {}
    
    def test_conversation_lifecycle(self):
        """Test the complete conversation lifecycle."""
        # Create a new conversation
        create_response = requests.post(
            f"{self.base_url}/conversations",
            headers=self.headers,
            json={"title": "API Test Conversation"}
        )
        
        assert create_response.status_code == 200
        conversation_data = create_response.json()
        conversation_id = conversation_data["conversation_id"]
        
        # Add a message to the conversation
        message_response = requests.post(
            f"{self.base_url}/conversations/{conversation_id}/messages",
            headers=self.headers,
            json={"content": "What is the derivative of x^3?"}
        )
        
        assert message_response.status_code == 200
        message_data = message_response.json()
        assert "message_id" in message_data
        assert "workflow_id" in message_data
        
        # Wait for the workflow to complete
        workflow_id = message_data["workflow_id"]
        max_wait_time = 15
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(
                f"{self.base_url}/workflows/{workflow_id}/status",
                headers=self.headers
            )
            
            assert status_response.status_code == 200
            status = status_response.json()
            
            if status["status"] in ["completed", "error"]:
                break
                
            time.sleep(0.5)
        
        assert status["status"] == "completed"
        
        # Get conversation history
        history_response = requests.get(
            f"{self.base_url}/conversations/{conversation_id}/messages",
            headers=self.headers
        )
        
        assert history_response.status_code == 200
        history = history_response.json()
        
        assert "messages" in history
        assert len(history["messages"]) == 2  # User message and system response
        
        # Check response content
        system_message = [m for m in history["messages"] if m.get("sender") == "system"][0]
        assert "3x^2" in system_message["content"]
        
        # Delete the conversation
        delete_response = requests.delete(
            f"{self.base_url}/conversations/{conversation_id}",
            headers=self.headers
        )
        
        assert delete_response.status_code == 200
    
    def test_mathematical_query_api(self):
        """Test the dedicated mathematical query API endpoint."""
        # Simple math query
        query_response = requests.post(
            f"{self.base_url}/math/query",
            headers=self.headers,
            json={
                "query": "Solve the equation 3x + 5 = 14 for x",
                "require_steps": True,
                "require_visualization": False
            }
        )
        
        assert query_response.status_code == 200
        result = query_response.json()
        
        assert "response" in result
        assert "x = 3" in result["response"]
        assert "steps" in result
        assert len(result["steps"]) > 0
        
        # More complex math query with visualization
        complex_query_response = requests.post(
            f"{self.base_url}/math/query",
            headers=self.headers,
            json={
                "query": "Plot the function f(x) = x^2 - 4x + 4 and identify its key features",
                "require_steps": True,
                "require_visualization": True
            }
        )
        
        assert complex_query_response.status_code == 200
        complex_result = complex_query_response.json()
        
        assert "response" in complex_result
        assert "parabola" in complex_result["response"].lower()
        assert "visualization_urls" in complex_result
        assert len(complex_result["visualization_urls"]) > 0
        
        # Verify visualization URL is accessible
        viz_url = complex_result["visualization_urls"][0]
        if viz_url.startswith("/"):
            viz_url = f"http://localhost:8000{viz_url}"
            
        viz_response = requests.get(viz_url)
        assert viz_response.status_code == 200
    
    def _create_handwritten_math_image(self):
        """Create a simple image with handwritten-like mathematical expression."""
        # Create a blank image with white background
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        
        # Use a font that looks somewhat like handwriting if available
        try:
            font = ImageFont.truetype("Arial", 40)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw a simple equation
        d.text((50, 70), "x^2 + 3x = 10", fill='black', font=font)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def test_handwriting_recognition_api(self):
        """Test the handwriting recognition API endpoint."""
        # Create a simple handwritten math image
        img_data = self._create_handwritten_math_image()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(img_data.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Upload the image for recognition
            with open(tmp_file_path, 'rb') as img_file:
                files = {'image': ('equation.png', img_file, 'image/png')}
                
                recognition_response = requests.post(
                    f"{self.base_url}/math/recognize",
                    headers=self.headers,
                    files=files
                )
            
            assert recognition_response.status_code == 200
            recognition_result = recognition_response.json()
            
            # Check recognition results
            assert "latex" in recognition_result
            assert "confidence" in recognition_result
            
            # The exact recognition will depend on the model, but it should contain key elements
            assert "x^2" in recognition_result["latex"]
            assert "3x" in recognition_result["latex"]
            assert "10" in recognition_result["latex"]
            assert recognition_result["confidence"] > 0.5
            
            # Test solving the recognized equation
            solve_response = requests.post(
                f"{self.base_url}/math/solve",
                headers=self.headers,
                json={
                    "latex": recognition_result["latex"],
                    "require_steps": True
                }
            )
            
            assert solve_response.status_code == 200
            solve_result = solve_response.json()
            
            assert "solution" in solve_result
            assert "steps" in solve_result
            assert len(solve_result["steps"]) > 0
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def test_visualization_api(self):
        """Test the dedicated visualization API endpoints."""
        # Test 2D function plotting
        plot_2d_response = requests.post(
            f"{self.base_url}/visualization/plot2d",
            headers=self.headers,
            json={
                "function": "sin(x) * cos(x)",
                "x_range": [-3.14, 3.14],
                "title": "Sin(x) * Cos(x)",
                "grid": True
            }
        )
        
        assert plot_2d_response.status_code == 200
        plot_2d_result = plot_2d_response.json()
        
        assert "visualization_url" in plot_2d_result
        assert "visualization_id" in plot_2d_result
        
        # Test 3D function plotting
        plot_3d_response = requests.post(
            f"{self.base_url}/visualization/plot3d",
            headers=self.headers,
            json={
                "function": "sin(sqrt(x^2 + y^2))",
                "x_range": [-5, 5],
                "y_range": [-5, 5],
                "title": "3D Sinc Function"
            }
        )
        
        assert plot_3d_response.status_code == 200
        plot_3d_result = plot_3d_response.json()
        
        assert "visualization_url" in plot_3d_result
        assert "visualization_id" in plot_3d_result
        
        # Test retrieving visualization metadata
        viz_id = plot_2d_result["visualization_id"]
        metadata_response = requests.get(
            f"{self.base_url}/visualization/{viz_id}/metadata",
            headers=self.headers
        )
        
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()
        
        assert "type" in metadata
        assert metadata["type"] == "plot2d"
        assert "function" in metadata
        assert metadata["function"] == "sin(x) * cos(x)"
    
    def test_latex_processing_api(self):
        """Test the LaTeX processing API endpoints."""
        # Test LaTeX to expression conversion
        latex_convert_response = requests.post(
            f"{self.base_url}/latex/convert",
            headers=self.headers,
            json={"latex": "\\int_{0}^{1} x^2 \\, dx"}
        )
        
        assert latex_convert_response.status_code == 200
        latex_result = latex_convert_response.json()
        
        assert "expression" in latex_result
        assert latex_result["expression"] == "Integral(x**2, (x, 0, 1))"
        
        # Test LaTeX rendering to image
        latex_render_response = requests.post(
            f"{self.base_url}/latex/render",
            headers=self.headers,
            json={"latex": "\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"}
        )
        
        assert latex_render_response.status_code == 200
        render_result = latex_render_response.json()
        
        assert "image_url" in render_result
        
        # Verify the image URL is accessible
        img_url = render_result["image_url"]
        if img_url.startswith("/"):
            img_url = f"http://localhost:8000{img_url}"
            
        img_response = requests.get(img_url)
        assert img_response.status_code == 200
    
    def test_search_integration_api(self):
        """Test the search integration API endpoint."""
        # Test mathematical search
        search_response = requests.post(
            f"{self.base_url}/search/math",
            headers=self.headers,
            json={
                "query": "Taylor series expansion",
                "max_results": 3
            }
        )
        
        assert search_response.status_code == 200
        search_results = search_response.json()
        
        assert "results" in search_results
        assert len(search_results["results"]) <= 3
        assert "sources" in search_results
        
        # Verify search result integration with computation
        integrated_response = requests.post(
            f"{self.base_url}/math/query-with-search",
            headers=self.headers,
            json={
                "query": "What is the Taylor series for sin(x)?",
                "include_search": True
            }
        )
        
        assert integrated_response.status_code == 200
        integrated_result = integrated_response.json()
        
        assert "response" in integrated_result
        assert "Taylor series" in integrated_result["response"]
        assert "sin(x)" in integrated_result["response"]
        assert "search_results" in integrated_result
        assert "sources" in integrated_result

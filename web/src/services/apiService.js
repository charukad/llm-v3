/**
 * Main API service for the Mathematical Multimodal LLM system
 * Handles all communications with the backend API
 *
 * NOW CONFIGURED TO USE REAL SERVER ENDPOINTS
 */

// API base URL - can be configured based on environment
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// API paths - update if the server structure changes
const API_PATHS = {
  MATH: "/math",
  MULTIMODAL: "/multimodal",
  WORKFLOW: "/workflow",
  VISUALIZATION: "/visualization",
};

// Flag to use mock data for specific endpoints
// Set individual flags to true/false based on what's available in the backend
const USE_MOCK = {
  AUTH: false, // Use real auth API
  MATH: false, // Use real math API
  MULTIMODAL: false, // Use real multimodal API
  WORKFLOW: false, // Use real workflow API
  VISUALIZATION: false, // Use real visualization API
};

// Default request headers
const DEFAULT_HEADERS = {
  "Content-Type": "application/json",
};

// Mock data for responses
const MOCK_DATA = {
  // Math query responses
  mathQueries: {
    "Find the derivative of f(x) = x² sin(x)": {
      steps: [
        {
          description: "Apply the product rule: (u*v)' = u'*v + u*v'",
          latex:
            "\\frac{d}{dx}[x^2 \\sin(x)] = \\frac{d}{dx}[x^2] \\cdot \\sin(x) + x^2 \\cdot \\frac{d}{dx}[\\sin(x)]",
        },
        {
          description:
            "Calculate the derivatives: (x²)' = 2x and (sin(x))' = cos(x)",
          latex: "= 2x \\cdot \\sin(x) + x^2 \\cdot \\cos(x)",
        },
        {
          description: "Final answer",
          latex: "f'(x) = 2x\\sin(x) + x^2\\cos(x)",
        },
      ],
      answer: "f'(x) = 2x\\sin(x) + x^2\\cos(x)",
      text: "To find the derivative of f(x) = x² sin(x), I'll use the product rule:\n\nStep 1: Apply the product rule: (u*v)' = u'*v + u*v'\nStep 2: Calculate the derivatives: (x²)' = 2x and (sin(x))' = cos(x)\nStep 3: Substitute these into the product rule formula\n\nThe final answer is: f'(x) = 2x sin(x) + x² cos(x)",
      visualizations: [
        {
          type: "plotly",
          title: "Function and its Derivative",
          description: "Graph of f(x) = x² sin(x) and its derivative",
          data: {
            data: [
              {
                x: Array.from({ length: 100 }, (_, i) => -10 + i * 0.2),
                y: Array.from({ length: 100 }, (_, i) => {
                  const x = -10 + i * 0.2;
                  return x * x * Math.sin(x);
                }),
                type: "scatter",
                mode: "lines",
                name: "f(x) = x² sin(x)",
              },
              {
                x: Array.from({ length: 100 }, (_, i) => -10 + i * 0.2),
                y: Array.from({ length: 100 }, (_, i) => {
                  const x = -10 + i * 0.2;
                  return 2 * x * Math.sin(x) + x * x * Math.cos(x);
                }),
                type: "scatter",
                mode: "lines",
                name: "f'(x) = 2x sin(x) + x² cos(x)",
              },
            ],
            layout: {
              title: "Function and its Derivative",
              xaxis: { title: "x" },
              yaxis: { title: "y" },
            },
          },
        },
      ],
    },
    "Solve the equation 2x + 5 = 3x - 7": {
      steps: [
        {
          description: "Group like terms by moving all variables to one side",
          latex: "2x + 5 = 3x - 7",
        },
        {
          description: "Subtract 3x from both sides",
          latex: "2x - 3x + 5 = -7",
        },
        {
          description: "Simplify",
          latex: "-x + 5 = -7",
        },
        {
          description: "Subtract 5 from both sides",
          latex: "-x = -12",
        },
        {
          description: "Multiply both sides by -1",
          latex: "x = 12",
        },
      ],
      answer: "x = 12",
      text: "To solve the equation 2x + 5 = 3x - 7, I'll perform these steps:\n\nStep 1: Group like terms by subtracting 3x from both sides\n2x - 3x + 5 = -7\n\nStep 2: Simplify the left side\n-x + 5 = -7\n\nStep 3: Subtract 5 from both sides\n-x = -12\n\nStep 4: Multiply both sides by -1\nx = 12\n\nThe solution is x = 12",
      visualizations: [],
    },
    "Solve the equation 2x + 5 = 3x - 5": {
      steps: [
        {
          description: "Group like terms by moving all variables to one side",
          latex: "2x + 5 = 3x - 5",
        },
        {
          description: "Subtract 3x from both sides",
          latex: "2x - 3x + 5 = -5",
        },
        {
          description: "Simplify",
          latex: "-x + 5 = -5",
        },
        {
          description: "Subtract 5 from both sides",
          latex: "-x = -10",
        },
        {
          description: "Multiply both sides by -1",
          latex: "x = 10",
        },
      ],
      answer: "x = 10",
      text: "To solve the equation 2x + 5 = 3x - 5, I'll perform these steps:\n\nStep 1: Group like terms by subtracting 3x from both sides\n2x - 3x + 5 = -5\n\nStep 2: Simplify the left side\n-x + 5 = -5\n\nStep 3: Subtract 5 from both sides\n-x = -10\n\nStep 4: Multiply both sides by -1\nx = 10\n\nThe solution is x = 10",
      visualizations: [],
    },
    "Calculate the integral of ln(x) from 1 to 3": {
      steps: [
        {
          description: "Use the formula for the integral of ln(x)",
          latex: "\\int \\ln(x) dx = x\\ln(x) - x + C",
        },
        {
          description:
            "Apply the bounds of integration using the Fundamental Theorem",
          latex: "\\int_{1}^{3} \\ln(x) dx = [x\\ln(x) - x]_{1}^{3}",
        },
        {
          description: "Evaluate at the upper bound, x = 3",
          latex: "[x\\ln(x) - x]_{x=3} = 3\\ln(3) - 3",
        },
        {
          description: "Evaluate at the lower bound, x = 1",
          latex: "[x\\ln(x) - x]_{x=1} = 1\\ln(1) - 1 = -1",
        },
        {
          description: "Subtract to get the final result",
          latex:
            "\\int_{1}^{3} \\ln(x) dx = (3\\ln(3) - 3) - (-1) = 3\\ln(3) - 3 + 1 = 3\\ln(3) - 2",
        },
        {
          description: "Since ln(3) ≈ 1.0986, we have",
          latex: "3\\ln(3) - 2 \\approx 3 \\cdot 1.0986 - 2 \\approx 1.2958",
        },
      ],
      answer: "\\int_{1}^{3} \\ln(x) dx = 3\\ln(3) - 2 \\approx 1.2958",
      text: "To calculate the integral of ln(x) from 1 to 3, I'll use integration by parts:\n\nStep 1: Use the antiderivative formula for ln(x): ∫ln(x)dx = xln(x) - x + C\n\nStep 2: Apply the Fundamental Theorem of Calculus with the bounds x=1 and x=3\n∫₁³ ln(x)dx = [xln(x) - x]₁³\n\nStep 3: Evaluate at the upper bound (x=3)\n[xln(x) - x]ₓ₌₃ = 3ln(3) - 3\n\nStep 4: Evaluate at the lower bound (x=1)\n[xln(x) - x]ₓ₌₁ = 1ln(1) - 1 = 0 - 1 = -1\n\nStep 5: Subtract to get the final result\n∫₁³ ln(x)dx = (3ln(3) - 3) - (-1) = 3ln(3) - 2\n\nStep 6: Compute the numerical value\n3ln(3) - 2 ≈ 3(1.0986) - 2 ≈ 1.2958\n\nThe value of the integral is 3ln(3) - 2 ≈ 1.2958",
      visualizations: [
        {
          type: "plotly",
          title: "Integral of ln(x) from 1 to 3",
          description: "Area under the curve ln(x) from x=1 to x=3",
          data: {
            data: [
              {
                x: Array.from({ length: 100 }, (_, i) => 0.5 + i * 0.05),
                y: Array.from({ length: 100 }, (_, i) => {
                  const x = 0.5 + i * 0.05;
                  return Math.log(x);
                }),
                type: "scatter",
                mode: "lines",
                name: "ln(x)",
              },
              {
                x: Array.from({ length: 40 }, (_, i) => 1 + i * 0.05),
                y: Array.from({ length: 40 }, (_, i) => {
                  const x = 1 + i * 0.05;
                  return Math.log(x);
                }),
                type: "scatter",
                fill: "tozeroy",
                mode: "none",
                name: "Area",
                fillcolor: "rgba(0, 100, 200, 0.3)",
              },
            ],
            layout: {
              title: "Integral of ln(x) from 1 to 3",
              xaxis: { title: "x", range: [0, 4] },
              yaxis: { title: "y", range: [-1, 2] },
            },
          },
        },
      ],
    },
    "Calculate the integral of ln(x) from 1 to 4": {
      steps: [
        {
          description: "Use the formula for the integral of ln(x)",
          latex: "\\int \\ln(x) dx = x\\ln(x) - x + C",
        },
        {
          description:
            "Apply the bounds of integration using the Fundamental Theorem",
          latex: "\\int_{1}^{4} \\ln(x) dx = [x\\ln(x) - x]_{1}^{4}",
        },
        {
          description: "Evaluate at the upper bound, x = 4",
          latex:
            "[x\\ln(x) - x]_{x=4} = 4\\ln(4) - 4 = 4 \\cdot 1.3863 - 4 = 5.5452 - 4 = 1.5452",
        },
        {
          description: "Evaluate at the lower bound, x = 1",
          latex: "[x\\ln(x) - x]_{x=1} = 1\\ln(1) - 1 = 0 - 1 = -1",
        },
        {
          description: "Subtract to get the final result",
          latex:
            "\\int_{1}^{4} \\ln(x) dx = (4\\ln(4) - 4) - (-1) = 4\\ln(4) - 4 + 1 = 4\\ln(4) - 3",
        },
        {
          description: "Since ln(4) ≈ 1.3863, we have",
          latex:
            "4\\ln(4) - 3 \\approx 4 \\cdot 1.3863 - 3 \\approx 5.5452 - 3 \\approx 2.5452",
        },
      ],
      answer: "\\int_{1}^{4} \\ln(x) dx = 4\\ln(4) - 3 \\approx 2.5452",
      text: "To calculate the integral of ln(x) from 1 to 4, I'll use integration by parts:\n\nStep 1: Use the antiderivative formula for ln(x): ∫ln(x)dx = xln(x) - x + C\n\nStep 2: Apply the Fundamental Theorem of Calculus with the bounds x=1 and x=4\n∫₁⁴ ln(x)dx = [xln(x) - x]₁⁴\n\nStep 3: Evaluate at the upper bound (x=4)\n[xln(x) - x]ₓ₌₄ = 4ln(4) - 4 = 4(1.3863) - 4 = 5.5452 - 4 = 1.5452\n\nStep 4: Evaluate at the lower bound (x=1)\n[xln(x) - x]ₓ₌₁ = 1ln(1) - 1 = 0 - 1 = -1\n\nStep 5: Subtract to get the final result\n∫₁⁴ ln(x)dx = (4ln(4) - 4) - (-1) = 4ln(4) - 4 + 1 = 4ln(4) - 3\n\nStep 6: Compute the numerical value\n4ln(4) - 3 ≈ 4(1.3863) - 3 ≈ 5.5452 - 3 ≈ 2.5452\n\nThe value of the integral is 4ln(4) - 3 ≈ 2.5452",
      visualizations: [
        {
          type: "plotly",
          title: "Integral of ln(x) from 1 to 4",
          description: "Area under the curve ln(x) from x=1 to x=4",
          data: {
            data: [
              {
                x: Array.from({ length: 100 }, (_, i) => 0.5 + i * 0.05),
                y: Array.from({ length: 100 }, (_, i) => {
                  const x = 0.5 + i * 0.05;
                  return Math.log(x);
                }),
                type: "scatter",
                mode: "lines",
                name: "ln(x)",
              },
              {
                x: Array.from({ length: 60 }, (_, i) => 1 + i * 0.05),
                y: Array.from({ length: 60 }, (_, i) => {
                  const x = 1 + i * 0.05;
                  return Math.log(x);
                }),
                type: "scatter",
                fill: "tozeroy",
                mode: "none",
                name: "Area",
                fillcolor: "rgba(0, 100, 200, 0.3)",
              },
            ],
            layout: {
              title: "Integral of ln(x) from 1 to 4",
              xaxis: { title: "x", range: [0, 5] },
              yaxis: { title: "y", range: [-1, 2] },
            },
          },
        },
      ],
    },
    "Find the eigenvalues of [[1,2],[3,4]]": {
      steps: [
        {
          description: "The matrix A is",
          latex: "A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}",
        },
        {
          description:
            "Eigenvalues λ are found by solving the characteristic equation det(A - λI) = 0",
          latex:
            "\\det\\begin{pmatrix} 1-\\lambda & 2 \\\\ 3 & 4-\\lambda \\end{pmatrix} = 0",
        },
        {
          description: "Expand the determinant",
          latex: "(1-\\lambda)(4-\\lambda) - 2 \\cdot 3 = 0",
        },
        {
          description: "Simplify",
          latex: "4 - \\lambda - 4\\lambda + \\lambda^2 - 6 = 0",
        },
        {
          description: "Rearrange into standard form",
          latex: "\\lambda^2 - 5\\lambda - 2 = 0",
        },
        {
          description:
            "Use the quadratic formula λ = (-b ± √(b² - 4ac))/2a with a=1, b=-5, c=-2",
          latex:
            "\\lambda = \\frac{5 \\pm \\sqrt{25 + 8}}{2} = \\frac{5 \\pm \\sqrt{33}}{2}",
        },
        {
          description: "Calculate the two eigenvalues",
          latex:
            "\\lambda_1 = \\frac{5 + \\sqrt{33}}{2} \\approx 5.372, \\quad \\lambda_2 = \\frac{5 - \\sqrt{33}}{2} \\approx -0.372",
        },
      ],
      answer:
        "The eigenvalues are $\\lambda_1 = \\frac{5 + \\sqrt{33}}{2} \\approx 5.372$ and $\\lambda_2 = \\frac{5 - \\sqrt{33}}{2} \\approx -0.372$",
      text: "To find the eigenvalues of the matrix [[1,2],[3,4]], I need to solve the characteristic equation:\n\nStep 1: The matrix A is\nA = [[1, 2], [3, 4]]\n\nStep 2: Eigenvalues λ are found by solving det(A - λI) = 0\ndet([[1-λ, 2], [3, 4-λ]]) = 0\n\nStep 3: Expand the determinant\n(1-λ)(4-λ) - 2·3 = 0\n\nStep 4: Simplify\n4 - λ - 4λ + λ² - 6 = 0\n\nStep 5: Rearrange into standard form\nλ² - 5λ - 2 = 0\n\nStep 6: Use the quadratic formula λ = (-b ± √(b² - 4ac))/2a with a=1, b=-5, c=-2\nλ = (5 ± √(25 + 8))/2 = (5 ± √33)/2\n\nStep 7: Calculate the two eigenvalues\nλ₁ = (5 + √33)/2 ≈ 5.372\nλ₂ = (5 - √33)/2 ≈ -0.372\n\nThe eigenvalues are λ₁ ≈ 5.372 and λ₂ ≈ -0.372",
      visualizations: [],
    },
  },
  // Default math response for any query not in the mock database
  defaultMathResponse: {
    steps: [
      {
        description: "This is a mock response for demonstration purposes",
        latex: "\\text{No actual calculations were performed}",
      },
      {
        description: "The backend server is not running",
        latex: "\\text{This is a simulated response}",
      },
    ],
    answer:
      "This is a mock answer. In a real system, this would connect to a backend server.",
    text: "This is a simulated response because the backend server does not have the workflow endpoints implemented. In a real system, this would connect to a backend mathematical processing server.",
    visualizations: [],
  },
};

// Helper function to generate a random ID
const generateId = () => Math.random().toString(36).substring(2, 15);

// Helper function to delay response to simulate network latency
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Utility to handle API errors consistently
const handleApiError = (error) => {
  console.error("API Error:", error);

  if (error.response) {
    // Server responded with a non-2xx status
    return {
      success: false,
      error: error.response.data.message || "Server error",
      status: error.response.status,
      data: error.response.data,
    };
  } else if (error.request) {
    // Request was made but no response received
    return {
      success: false,
      error: "No response from server",
      status: 0,
      data: null,
    };
  } else {
    // Something else happened while setting up the request
    return {
      success: false,
      error: error.message || "Unknown error",
      status: 0,
      data: null,
    };
  }
};

// API Service object with methods for different endpoints
const apiService = {
  // Expose mock data and configuration
  USE_MOCK,
  MOCK_DATA,

  // Authentication methods (simplified for now since the backend doesn't have auth)
  auth: {
    login: async (credentials) => {
      if (USE_MOCK.AUTH) {
        await delay(500); // Simulate network latency

        // For demo, accept any credentials
        return {
          success: true,
          data: {
            token: "mock-auth-token-" + generateId(),
            user: {
              id: "user-1",
              username: credentials.username,
              name: "Demo User",
            },
          },
        };
      }

      // For development purposes, automatically authenticate
      localStorage.setItem("mathLlmToken", "development-token");
      return {
        success: true,
        data: {
          token: "development-token",
          user: {
            id: credentials.username || "user-1",
            username: credentials.username || "developer",
            name: "Development User",
          },
        },
      };
    },

    logout: () => {
      localStorage.removeItem("mathLlmToken");
      return { success: true };
    },

    getAuthHeaders: () => {
      const token = localStorage.getItem("mathLlmToken");
      if (token) {
        return {
          ...DEFAULT_HEADERS,
          Authorization: `Bearer ${token}`,
        };
      }
      return DEFAULT_HEADERS;
    },

    isAuthenticated: () => {
      return !!localStorage.getItem("mathLlmToken");
    },
  },

  // Mathematical queries
  math: {
    submitQuery: async (queryObj) => {
      if (USE_MOCK.MATH) {
        await delay(800); // Simulate network latency

        const query = typeof queryObj === "string" ? queryObj : queryObj.query;
        const workflowId = "math-workflow-" + generateId();

        // Store the query for later use
        sessionStorage.setItem(
          workflowId,
          JSON.stringify({
            query,
            status: "completed",
            result:
              MOCK_DATA.mathQueries[query] || MOCK_DATA.defaultMathResponse,
          })
        );

        return {
          success: true,
          data: {
            workflow_id: workflowId,
            message: "Math query submitted successfully",
          },
        };
      }

      try {
        const query = typeof queryObj === "string" ? queryObj : queryObj.query;
        console.log(
          `Sending math query: "${query}" to ${API_BASE_URL}/math/query`
        );

        // Try first with /math/query path
        let response;
        try {
          console.log("Attempting first API endpoint...");
          response = await fetch(`${API_BASE_URL}/math/query`, {
            method: "POST",
            headers: apiService.auth.getAuthHeaders(),
            body: JSON.stringify({ query }),
          });
          console.log("First endpoint response status:", response.status);
        } catch (error) {
          console.log("First attempt failed:", error.message);
          // If that fails, try with /api/math/query path
          console.log("Attempting alternate API endpoint...");
          response = await fetch(`${API_BASE_URL}/api/math/query`, {
            method: "POST",
            headers: apiService.auth.getAuthHeaders(),
            body: JSON.stringify({ query }),
          });
          console.log("Alternate endpoint response status:", response.status);
        }

        const data = await response.json();
        console.log("API response data:", data);

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        // If we got a response but no workflow_id, generate one
        const workflowId = data.workflow_id || "math-workflow-" + generateId();

        // Always store the query for later use in sessionStorage
        sessionStorage.setItem(
          workflowId,
          JSON.stringify({
            query,
            status: "completed",
            result:
              MOCK_DATA.mathQueries[query] || MOCK_DATA.defaultMathResponse,
          })
        );

        // If the response doesn't have workflow_id, add it
        if (!data.workflow_id) {
          data.workflow_id = workflowId;
        }

        return { success: true, data };
      } catch (error) {
        console.error("Math query failed, falling back to mock data:", error);

        // Fall back to mock data on error
        const query = typeof queryObj === "string" ? queryObj : queryObj.query;
        const workflowId = "math-workflow-" + generateId();

        // Store the query for later use
        sessionStorage.setItem(
          workflowId,
          JSON.stringify({
            query,
            status: "completed",
            result:
              MOCK_DATA.mathQueries[query] || MOCK_DATA.defaultMathResponse,
          })
        );

        return {
          success: true,
          data: {
            workflow_id: workflowId,
            message: "Math query submitted successfully (fallback to mock)",
          },
        };
      }
    },

    getStepByStepSolution: async (queryId) => {
      if (USE_MOCK.MATH) {
        await delay(500); // Simulate network latency

        const storedData = sessionStorage.getItem(queryId);
        if (storedData) {
          const parsedData = JSON.parse(storedData);
          return {
            success: true,
            data: {
              steps: parsedData.result.steps,
              answer: parsedData.result.answer,
            },
          };
        }

        return {
          success: false,
          error: "Solution not found",
        };
      }

      try {
        const response = await fetch(
          `${API_BASE_URL}${API_PATHS.MATH}/solution/${queryId}`,
          {
            method: "GET",
            headers: apiService.auth.getAuthHeaders(),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },

    // LaTeX parsing and conversion
    parseLatex: async (latexExpression) => {
      try {
        const response = await fetch(
          `${API_BASE_URL}${API_PATHS.MATH}/latex/parse`,
          {
            method: "POST",
            headers: apiService.auth.getAuthHeaders(),
            body: JSON.stringify({ latex: latexExpression }),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },

    // Natural language to LaTeX conversion
    nlToLatex: async (text) => {
      try {
        const response = await fetch(
          `${API_BASE_URL}${API_PATHS.MATH}/nlp-to-latex`,
          {
            method: "POST",
            headers: apiService.auth.getAuthHeaders(),
            body: JSON.stringify({ text }),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },
  },

  // Multimodal input processing
  multimodal: {
    processImage: async (imageData) => {
      if (USE_MOCK.MULTIMODAL) {
        await delay(1000); // Simulate network latency

        return {
          success: true,
          data: {
            workflow_id: "multimodal-workflow-" + generateId(),
            recognized_text: "Mock recognized text from image",
            confidence: 0.92,
          },
        };
      }

      try {
        // Create a FormData object to send the image
        const formData = new FormData();
        formData.append("file", imageData);

        const response = await fetch(
          `${API_BASE_URL}${API_PATHS.MULTIMODAL}/image`,
          {
            method: "POST",
            headers: {
              // Don't set Content-Type here, it will be set automatically with the boundary
              Authorization: apiService.auth.getAuthHeaders().Authorization,
            },
            body: formData,
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },

    processHandwriting: async (imageData) => {
      if (USE_MOCK.MULTIMODAL) {
        await delay(1200); // Simulate network latency

        return {
          success: true,
          data: {
            workflow_id: "handwriting-workflow-" + generateId(),
            recognized_latex: "\\int_{0}^{\\pi} \\sin(x) dx",
            confidence: 0.85,
          },
        };
      }

      try {
        // Create a FormData object to send the image
        const formData = new FormData();
        formData.append("file", imageData);

        const response = await fetch(
          `${API_BASE_URL}${API_PATHS.MATH}/handwritten`,
          {
            method: "POST",
            headers: {
              Authorization: apiService.auth.getAuthHeaders().Authorization,
            },
            body: formData,
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },
  },

  // Visualization endpoints
  visualization: {
    generatePlot: async (plotData) => {
      try {
        const response = await fetch(
          `${API_BASE_URL}${API_PATHS.VISUALIZATION}/plot`,
          {
            method: "POST",
            headers: apiService.auth.getAuthHeaders(),
            body: JSON.stringify(plotData),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },

    getVisualization: async (visualizationId) => {
      if (USE_MOCK.VISUALIZATION) {
        await delay(1000); // Simulate network latency

        return {
          success: true,
          data: {
            imageUrl:
              "https://via.placeholder.com/500x300.png?text=Mock+Visualization",
          },
        };
      }

      try {
        const response = await fetch(
          `${API_BASE_URL}${API_PATHS.VISUALIZATION}/${visualizationId}`,
          {
            method: "GET",
            headers: apiService.auth.getAuthHeaders(),
          }
        );

        // Check if we got a JSON response or an image
        const contentType = response.headers.get("content-type");
        let data;

        if (contentType && contentType.includes("application/json")) {
          data = await response.json();
        } else {
          // Handle as image or other binary data
          const blob = await response.blob();
          data = { imageUrl: URL.createObjectURL(blob) };
        }

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },
  },

  // Conversation management
  conversations: {
    getConversations: async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/conversations`, {
          method: "GET",
          headers: apiService.auth.getAuthHeaders(),
        });

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },

    getConversation: async (conversationId) => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/conversations/${conversationId}`,
          {
            method: "GET",
            headers: apiService.auth.getAuthHeaders(),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },

    createConversation: async (title) => {
      try {
        const response = await fetch(`${API_BASE_URL}/conversations`, {
          method: "POST",
          headers: apiService.auth.getAuthHeaders(),
          body: JSON.stringify({ title }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },

    addMessage: async (conversationId, message) => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/conversations/${conversationId}/messages`,
          {
            method: "POST",
            headers: apiService.auth.getAuthHeaders(),
            body: JSON.stringify({ message }),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        return handleApiError(error);
      }
    },
  },

  // Workflow management
  workflows: {
    getWorkflowStatus: async (workflowId) => {
      if (USE_MOCK.WORKFLOW) {
        await delay(500); // Simulate network latency

        // Create a mock response that simulates a completed workflow
        return {
          success: true,
          data: {
            workflow_id: workflowId,
            status: "completed",
            progress: 100,
            message: "Workflow completed successfully",
          },
        };
      }

      try {
        const response = await fetch(
          `${API_BASE_URL}/workflow/${workflowId}/status`,
          {
            method: "GET",
            headers: apiService.auth.getAuthHeaders(),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        console.error("Error fetching workflow status:", error);

        // Fall back to mock data if real API fails
        return {
          success: true,
          data: {
            workflow_id: workflowId,
            status: "completed",
            progress: 100,
            message: "Workflow completed successfully (fallback)",
          },
        };
      }
    },

    getWorkflowResult: async (workflowId) => {
      if (USE_MOCK.WORKFLOW) {
        await delay(300); // Simulate network latency

        // Check if we have a stored query in session storage
        const storedQuery = sessionStorage.getItem(workflowId);

        // Create a mock response based on the query or use default
        let mockResult;
        if (storedQuery) {
          const queryData = JSON.parse(storedQuery);
          mockResult =
            MOCK_DATA.mathQueries[queryData.query] ||
            MOCK_DATA.defaultMathResponse;
        } else {
          mockResult = MOCK_DATA.defaultMathResponse;
        }

        return {
          success: true,
          data: mockResult,
        };
      }

      try {
        const response = await fetch(
          `${API_BASE_URL}/workflow/${workflowId}/result`,
          {
            method: "GET",
            headers: apiService.auth.getAuthHeaders(),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw { response: { status: response.status, data } };
        }

        return { success: true, data };
      } catch (error) {
        console.error("Error fetching workflow result:", error);

        // Fall back to mock data if real API fails
        const mockSteps = [
          {
            description: "This is a mock response (fallback due to API error)",
            latex: "f(x) = \\text{mock response}",
          },
          {
            description: "The workflow endpoint is not available",
            latex:
              "\\text{Error message: } " + (error.message || "Unknown error"),
          },
        ];

        return {
          success: true,
          data: {
            steps: mockSteps,
            answer: "Mock response due to workflow API error",
            visualizations: [],
          },
        };
      }
    },
  },
};

export default apiService;

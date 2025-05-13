#!/bin/bash

# Test script for the central input agent endpoint
# This script sends various requests to test the central processing endpoint

# Set the base URL
BASE_URL="http://localhost:8000"
ENDPOINT="/multimodal/central/process"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Header
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Testing Central Input Agent Endpoint${NC}"
echo -e "${BLUE}============================================================${NC}"

# Test health endpoint
echo -e "\n${BLUE}Testing health endpoint...${NC}"
curl -s "${BASE_URL}/multimodal/central/health" | python -m json.tool

# Function to make a POST request to the central endpoint
function test_request() {
    local name=$1
    local data=$2
    
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}Testing: $name${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo -e "Request data: $data\n"
    
    response=$(curl -s -X POST "${BASE_URL}${ENDPOINT}" \
        -H "Content-Type: application/json" \
        -d "$data")
    
    echo -e "Response:\n"
    echo $response | python -m json.tool
    
    # Check if the request was successful
    if echo $response | grep -q '"success": true'; then
        echo -e "\n${GREEN}✓ Request successful${NC}"
    else
        echo -e "\n${RED}✗ Request failed${NC}"
    fi
    
    echo -e "${BLUE}------------------------------------------------------------${NC}"
}

# Test 1: Visualization request
echo -e "\n${BLUE}Test 1: Visualization Request${NC}"
test_request "Visualization Request" '{
    "input_type": "text",
    "content": "Create a bar chart showing the monthly sales data for Q1 2023",
    "conversation_id": "test-conversation-vis",
    "parameters": {
        "prefer_interactive": true,
        "color_scheme": "viridis"
    }
}'

# Test 2: Mathematical computation request
echo -e "\n${BLUE}Test 2: Mathematical Computation Request${NC}"
test_request "Math Computation" '{
    "input_type": "text",
    "content": "Find the derivative of f(x) = x^3 * sin(x)",
    "conversation_id": "test-conversation-math",
    "parameters": {
        "show_steps": true,
        "format": "latex"
    }
}'

# Test 3: General knowledge query
echo -e "\n${BLUE}Test 3: General Knowledge Query${NC}"
test_request "General Knowledge" '{
    "input_type": "text",
    "content": "Explain the concept of machine learning and its main approaches",
    "conversation_id": "test-conversation-general",
    "parameters": {
        "detail_level": "intermediate"
    }
}'

# Test 4: Explicit agent routing
echo -e "\n${BLUE}Test 4: Explicit Agent Routing${NC}"
test_request "Explicit Routing to Search" '{
    "input_type": "text",
    "content": "What are the latest developments in quantum computing?",
    "conversation_id": "test-conversation-search",
    "target_agent": "search",
    "parameters": {
        "source_preference": "academic"
    }
}'

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}Testing complete${NC}"
echo -e "${BLUE}============================================================${NC}" 
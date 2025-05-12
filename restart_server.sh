#!/bin/bash
# Script to gracefully restart the server

echo "Finding current server process..."
SERVER_PID=$(ps aux | grep "python run_server.py" | grep -v grep | awk '{print $2}')

if [ -n "$SERVER_PID" ]; then
    echo "Stopping server process with PID $SERVER_PID..."
    kill $SERVER_PID
    echo "Waiting for server to stop..."
    sleep 3
    
    # Check if process is still running
    if ps -p $SERVER_PID > /dev/null; then
        echo "Server didn't stop gracefully, forcing termination..."
        kill -9 $SERVER_PID
        sleep 2
    fi
else
    echo "No running server found."
fi

# Clear any remaining Python processes that might be related
echo "Checking for any related Python processes..."
PYTHON_PIDS=$(ps aux | grep "python" | grep -v grep | grep -v "restart_server" | awk '{print $2}')
if [ -n "$PYTHON_PIDS" ]; then
    echo "Found other Python processes that might be related. Please check manually:"
    ps aux | grep "python" | grep -v grep | grep -v "restart_server"
fi

# Clear any old log files that might be keeping file handles open
echo "Backing up existing logs..."
mkdir -p logs/backup
cp logs/*.log logs/backup/ 2>/dev/null

# Start the server with improved settings
echo "Starting server with improved settings..."
export PYTHONUNBUFFERED=1
export MAX_RESPONSE_TIMEOUT=30  # 30 second timeout for responses

# Configure GPU and layer settings
export USE_MPS=1
export MODEL_LAYERS=128
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ENABLE_INFERENCE_FASTPATH=1

python run_server.py --port 8000 &

echo "Waiting for server to initialize..."
sleep 10

# Test if server is responding
echo "Testing server connection..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health

if [ $? -eq 0 ]; then
    echo "Server restarted successfully!"
else
    echo "Server might not have started correctly. Check logs for details."
fi

echo "Done! You can now try your API calls again." 
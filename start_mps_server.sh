#!/bin/bash
# MPS-accelerated server start script
export USE_MPS=1
export MODEL_LAYERS=128
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ENABLE_INFERENCE_FASTPATH=1

# Start the server on port 8000 to avoid conflicts
python run_server.py --port 8000 
#!/usr/bin/env bash
# ==============================================================================
# Creaky-CV5 · 1-Command AWS Graviton4 + OpenCV COOL Deployment Script
# ==============================================================================
# Deploys the Creaky-CV5 FastAPI + HTML5 <canvas> Cockpit on an AWS EC2
# Graviton instance running the official OpenCV COOL (Cloud-Optimized OpenCV
# Library / OpenCV 5 + KleidiCV) AMI from the AWS Marketplace.
#
# Usage on the AWS EC2 COOL instance (Ubuntu 24.04 ARM64 / Graviton c7g.xlarge):
#   bash opencv_hackathon/deploy_aws_cool.sh
# ==============================================================================
set -euo pipefail

echo "=================================================================="
echo " Creaky-CV5 · AWS COOL (OpenCV 5 + KleidiCV) Deployment Bootstrap"
echo "=================================================================="

python3 -c "import cv2, platform; print('Detected OpenCV:', cv2.__version__, 'on', platform.machine())"

if ! command -v uv >/dev/null 2>&1; then
    echo "[1/3] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[2/3] Installing FastAPI + Uvicorn + ONNX runtime..."
uv pip install --system fastapi uvicorn numpy pillow pydicom onnx torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

echo "[3/3] Exporting static fallback bundle & launching Creaky-CV5 Cockpit on :8055..."
python3 -m opencv_hackathon.app --export-static
exec python3 -m opencv_hackathon.app

#!/bin/bash
echo "============================================"
echo "   Durian Maturity Assessment App - Setup"
echo "============================================"
echo ""

# System dependencies (Raspberry Pi OS)
echo "[1/3] Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-tk \
    python3-pip \
    python3-pil \
    python3-pil.imagetk \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7 \
    libhdf5-dev

echo ""
echo "[2/3] Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "[3/3] Setup complete!"
echo "Run the app with:  python3 main.py"
echo "============================================"

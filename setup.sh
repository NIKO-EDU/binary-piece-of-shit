#!/usr/bin/env bash
# setup.sh — Create venv and install all SOD-OPT dependencies.
#
# Usage:
#   bash setup.sh              # auto-detect CUDA (falls back to CPU)
#   bash setup.sh cpu          # force CPU-only PyTorch
#   bash setup.sh cu121        # force CUDA 12.1 build
#   bash setup.sh cu118        # force CUDA 11.8 build
#
# After setup:
#   source venv/bin/activate
#   make                       # build the C kernel (requires gcc + libgomp)
#   python scripts/train_bnn.py --device cuda --epochs 20
#
# Dataset:
#   Place VisDrone2019-DET-train/ under data/ before training.
#   Download from: https://github.com/VisDrone/VisDrone-Dataset
#
# YOLOv8 weights (needed for benchmark_yolo.py only):
#   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
#   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

set -euo pipefail

CUDA_ARG="${1:-auto}"

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[setup] venv already exists, skipping creation."
fi

source venv/bin/activate
pip install --quiet --upgrade pip

# ── 2. Detect CUDA if auto ────────────────────────────────────────────────────
if [ "$CUDA_ARG" = "auto" ]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
        if [ "$MAJOR" -ge 12 ]; then
            CUDA_ARG="cu121"
        elif [ "$MAJOR" -eq 11 ] && [ "$MINOR" -ge 8 ]; then
            CUDA_ARG="cu118"
        else
            CUDA_ARG="cpu"
        fi
        echo "[setup] Detected CUDA ${CUDA_VERSION} → using ${CUDA_ARG} build"
    elif command -v nvidia-smi &>/dev/null; then
        # nvcc not on PATH but driver present
        CUDA_ARG="cu121"
        echo "[setup] nvidia-smi found, assuming CUDA 12.x → using ${CUDA_ARG} build"
    else
        CUDA_ARG="cpu"
        echo "[setup] No CUDA detected → using CPU-only PyTorch"
    fi
fi

# ── 3. Install PyTorch ────────────────────────────────────────────────────────
if [ "$CUDA_ARG" = "cpu" ]; then
    echo "[setup] Installing CPU-only PyTorch..."
    pip install torch torchvision
else
    echo "[setup] Installing PyTorch with ${CUDA_ARG}..."
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_ARG}"
fi

# ── 4. Install remaining dependencies ────────────────────────────────────────
echo "[setup] Installing remaining dependencies from requirements.txt..."
# torch/torchvision already installed — skip those lines to avoid re-downloading
grep -v '^torch' requirements.txt | grep -v '^#' | grep -v '^$' | pip install --quiet -r /dev/stdin

# ── 5. Build C kernel ─────────────────────────────────────────────────────────
echo "[setup] Building C kernel (make)..."
if command -v gcc &>/dev/null; then
    make
    echo "[setup] kernel.so built."
else
    echo "[setup] WARNING: gcc not found. Run 'make' manually after installing gcc."
    echo "         On Ubuntu: sudo apt install gcc libomp-dev"
fi

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo ""
echo "  # Download dataset (if not already present):"
echo "  #   https://github.com/VisDrone/VisDrone-Dataset"
echo "  #   → extract to data/VisDrone2019-DET-train/"
echo ""
echo "  # (Optional) Download YOLOv8 weights for benchmarking:"
echo "  wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
echo "  wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
echo ""
echo "  # Train:"
echo "  python scripts/train_bnn.py --device cuda --epochs 20"
echo "  python scripts/train_bnn.py --device cuda --epochs 20 --max-samples 100  # quick smoke-test"

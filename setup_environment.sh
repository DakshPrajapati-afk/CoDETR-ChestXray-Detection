#!/bin/bash
# Setup script for Co-DETR training on VinBigData dataset
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "=========================================="
echo "Co-DETR Environment Setup for VinBigData"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d "Co-DETR" ]; then
    echo "Error: Co-DETR directory not found. Please run this script from /home/dpraja12/Dino"
    exit 1
fi

# Load CUDA module if available (helps with MMCV compilation)
if command -v module &> /dev/null; then
    echo ""
    echo "Loading CUDA module..."
    module load cuda-11.3.1-gcc-12.1.0 2>/dev/null || echo "Note: Could not load CUDA module, continuing anyway"
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv_codetr

# Activate virtual environment
echo "Activating virtual environment..."
source venv_codetr/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch 1.12.1 with CUDA 11.3..."
# PyTorch 1.12.1 is compatible with Co-DETR and more readily available
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install MMCV
echo ""
echo "Installing MMCV 1.5.0..."
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# Install other dependencies
echo ""
echo "Installing other dependencies..."
cd Co-DETR
pip install -r requirements/build.txt
pip install -r requirements/runtime.txt

# Install Co-DETR using legacy setup.py method (most reliable)
echo ""
echo "Installing Co-DETR (this may take a few minutes)..."
python setup.py develop

# Install additional useful packages
pip install tensorboard
pip install pandas
pip install Pillow
pip install tqdm

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv_codetr/bin/activate"
echo ""
echo "Next steps:"
echo "1. Convert the dataset: python convert_vinbigdata_to_coco.py"
echo "2. Start training: bash train_vinbigdata.sh"

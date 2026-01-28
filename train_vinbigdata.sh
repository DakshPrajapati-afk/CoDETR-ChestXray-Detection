#!/bin/bash
# Training script for Co-DETR on VinBigData dataset
# Usage: bash train_vinbigdata.sh [num_gpus] [config_file]

set -e

# Default parameters
NUM_GPUS=${1:-2}  # Default to 2 GPUs
CONFIG=${2:-"Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_vinbigdata_dicom.py"}  # DICOM config
WORK_DIR="work_dirs/$(basename $CONFIG .py)"

echo "=========================================="
echo "Co-DETR Training on VinBigData"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Work Dir: $WORK_DIR"
echo "=========================================="

# Load CUDA module if available
if command -v module &> /dev/null; then
    module load cuda-11.3.1-gcc-12.1.0 2>/dev/null || echo "Note: CUDA module not loaded (may already be available)"
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Activating virtual environment..."
    if [ -d "venv_codetr" ]; then
        source venv_codetr/bin/activate
    else
        echo "Trying conda environment..."
        conda activate codetr 2>/dev/null || echo "Warning: No environment activated"
    fi
fi

# Check if dataset has been converted
if [ ! -f "/scratch/dpraja12/data/VinBigData/annotations/instances_train.json" ]; then
    echo "Error: Dataset not converted to COCO format!"
    echo "Please run: python convert_vinbigdata_to_coco.py"
    exit 1
fi

# Create work directory
mkdir -p $WORK_DIR

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on available GPUs

# Training command
echo ""
echo "Starting training..."
echo ""

cd Co-DETR

# Use distributed training for multi-GPU
if [ $NUM_GPUS -gt 1 ]; then
    bash tools/dist_train.sh ../$CONFIG $NUM_GPUS ../$WORK_DIR
else
    python3 tools/train.py ../$CONFIG --work-dir ../$WORK_DIR
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Results saved to: $WORK_DIR"

#!/bin/bash
# Resume training script for Co-DETR on VinBigData dataset
# Usage: bash resume_training.sh [num_gpus] [checkpoint]

set -e

# Default parameters
NUM_GPUS=${1:-2}  # Default to 2 GPUs
CONFIG="Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_vinbigdata_dicom.py"
WORK_DIR="work_dirs/co_dino_5scale_r50_vinbigdata_dicom"
CHECKPOINT=${2:-"$WORK_DIR/latest.pth"}

echo "=========================================="
echo "Co-DETR Training Resume on VinBigData"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Work Dir: $WORK_DIR"
echo "Checkpoint: $CHECKPOINT"
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

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la $WORK_DIR/*.pth 2>/dev/null || echo "No checkpoints found in $WORK_DIR"
    exit 1
fi

# Create work directory if needed
mkdir -p $WORK_DIR

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on available GPUs

# Training command
echo ""
echo "Resuming training from checkpoint..."
echo ""

cd Co-DETR

# Use distributed training for multi-GPU
if [ $NUM_GPUS -gt 1 ]; then
    python3 -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        tools/train.py \
        ../$CONFIG \
        --work-dir ../$WORK_DIR \
        --resume-from ../$CHECKPOINT \
        --launcher pytorch
else
    python3 tools/train.py ../$CONFIG --work-dir ../$WORK_DIR --resume-from ../$CHECKPOINT
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Results saved to: $WORK_DIR"

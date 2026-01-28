#!/bin/bash
# Evaluation script for Co-DETR on VinBigData dataset
# Usage: bash evaluate_model.sh [checkpoint_path] [config_file]

set -e

# Parameters
CHECKPOINT=${1:-"work_dirs/co_dino_5scale_r50_vinbigdata/latest.pth"}
CONFIG=${2:-"Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_vinbigdata.py"}

echo "=========================================="
echo "Co-DETR Evaluation on VinBigData"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "=========================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv_codetr/bin/activate
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo ""
echo "Running evaluation..."
echo ""

cd Co-DETR

python tools/test.py \
    ../$CONFIG \
    ../$CHECKPOINT \
    --eval bbox \
    --show-dir ../eval_results

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
echo "Results saved to: eval_results/"

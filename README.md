# Co-DETR for VinBigData Chest X-Ray Abnormality Detection

This repository contains an implementation of **Co-DETR (Collaborative DETR)** for detecting 14 types of chest X-ray abnormalities using the VinBigData dataset.

## Results

### Detection Performance (Epoch 44)

| Metric | Value |
|--------|-------|
| mAP | 0.453 |
| mAP@50 | 0.741 |
| mAP@75 | 0.477 |
| AR@100 | 0.709 |

### One Detection Per Disease Mode

Added a `one_detection_per_class` option that keeps only the highest-scoring detection per disease class:

| Metric | Multiple Det/Disease | One Det/Disease |
|--------|---------------------|-----------------|
| mAP | 0.453 | 0.243 |
| mAP@50 | 0.741 | 0.396 |

## Sample Predictions

### Best Predictions
| Rank 1 | Rank 2 |
|--------|--------|
| ![Best 1](results/best/01_a040343977edb13d15604e5e3c125e59.png) | ![Best 2](results/best/02_bd3fa1499eba12696a695b952f9218a3.png) |

### Worst Predictions
| Rank 1 | Rank 2 |
|--------|--------|
| ![Worst 1](results/worst/01_9ea6802588a0cf35c44631bf01f9bd7f.png) | ![Worst 2](results/worst/02_4c02da7cd2dc7415b226103114e5aaf0.png) |

## Dataset Information

- **Dataset**: VinBigData Chest X-ray Abnormalities Detection
- **Training Images**: 15,000
- **Test Images**: 3,000
- **Classes**: 14 abnormality types
  - Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation
  - ILD, Infiltration, Lung Opacity, Nodule/Mass, Other lesion
  - Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv_codetr
source venv_codetr/bin/activate

# Install dependencies
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.0
cd Co-DETR && pip install -e .
```

### 2. Training

```bash
cd Co-DETR
python tools/train.py projects/configs/co_dino/co_dino_5scale_r50_vinbigdata_dicom.py
```

### 3. Evaluation

```bash
python tools/test.py \
    projects/configs/co_dino/co_dino_5scale_r50_vinbigdata_dicom.py \
    work_dirs/co_dino_5scale_r50_vinbigdata_dicom/epoch_44.pth \
    --eval bbox
```

### 4. Visualization

```bash
python visualize_predictions.py \
    --results work_dirs/co_dino_5scale_r50_vinbigdata_dicom/results.pkl \
    --ann /path/to/annotations/instances_val.json \
    --img-prefix /path/to/images/ \
    --output results/
```

## Repository Structure

```
.
├── Co-DETR/                                    # Main Co-DETR model code
│   ├── projects/
│   │   ├── configs/
│   │   │   ├── _base_/datasets/
│   │   │   │   ├── vinbigdata_detection.py     # Dataset config (PNG)
│   │   │   │   └── vinbigdata_dicom.py         # Dataset config (DICOM)
│   │   │   └── co_dino/
│   │   │       ├── co_dino_5scale_r50_vinbigdata.py
│   │   │       └── co_dino_5scale_r50_vinbigdata_dicom.py
│   │   └── models/
│   │       ├── co_deformable_detr_head.py      # Detection head (one_detection_per_class)
│   │       ├── co_detr.py                      # Main model
│   │       └── co_dino_head.py
│   └── tools/
│       ├── train.py
│       └── test.py
├── results/                                    # Visualization results
│   ├── best/                                   # Top performing predictions
│   ├── worst/                                  # Poorest performing predictions
│   └── random/                                 # Random sample predictions
├── visualize_predictions.py                    # Visualization script
└── README.md
```

## Key Files

| File | Description |
|------|-------------|
| `Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_vinbigdata_dicom.py` | Main model configuration |
| `Co-DETR/projects/models/co_deformable_detr_head.py` | Detection head with `one_detection_per_class` |
| `visualize_predictions.py` | Script to generate prediction visualizations |

## Configuration Options

Enable one detection per disease in `test_cfg`:
```python
test_cfg=[
    dict(
        max_per_img=300,
        one_detection_per_class=True,  # Only keep best detection per disease
    ),
    ...
]
```

## Model Configurations

### Co-DINO ResNet-50 (Recommended for Fast Training)
- **Config**: `Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_vinbigdata.py`
- **Expected Performance**: ~0.75-0.80 mAP@0.5
- **Epoch Time**: ~15-20 minutes (2 A100 GPUs)
- **Total Training Time**: ~9-12 hours (36 epochs)
- **GPU Memory**: ~12-14 GB per GPU

### Training Parameters
- **Optimizer**: AdamW (lr=2e-4, weight_decay=0.0001)
- **Batch Size**: 4 per GPU (total 8 with 2 GPUs)
- **Image Size**: 1024x1024
- **Epochs**: 36
- **LR Schedule**: Step decay at epochs 20 and 28
- **Mixed Precision**: FP16 enabled for faster training

## Resource Recommendations

### For Efficient Training (Target ~0.80 mAP)

**Recommended Configuration:**
- **GPUs**: 2x A100 (40GB or 80GB)
- **CPU Cores**: 8-12 cores
- **RAM**: 32 GB
- **Batch Size**: 4 per GPU (total batch size: 8)
- **Workers**: 4 per GPU

**Expected Performance:**
- Epoch time: ~15-20 minutes
- Total training (36 epochs): ~10-12 hours
- GPU utilization: ~85-95%
- GPU memory: ~12-14 GB per GPU with mixed precision

### For Faster Experimentation (Lower epochs)

If you want to quickly test:
```bash
# Modify the config to train for 12 epochs instead of 36
# Edit: Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_vinbigdata.py
# Line: runner = dict(type='EpochBasedRunner', max_epochs=12)
```

This will give you reasonable results (~0.70-0.75 mAP) in ~3-4 hours.

### For Better Performance (Larger Model)

To achieve ~0.85 mAP, you would need:
1. Longer training (48-60 epochs)
2. Potentially a larger backbone (Swin-Tiny or Swin-Small)
3. More data augmentation
4. Model ensemble

**Note**: Medical imaging tasks typically have lower mAP scores compared to natural images. An mAP@0.5 of 0.80-0.85 would be excellent for this dataset.

## SLURM Job Submission

If you're using SLURM on ASU Sol:

```bash
#!/bin/bash
#SBATCH --job-name=codetr_vinbig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules if needed
module load cuda/11.3

# Activate environment
cd /home/dpraja12/Dino
source venv_codetr/bin/activate

# Run training
bash train_vinbigdata.sh 2
```

Save as `submit_training.slurm` and submit with:
```bash
sbatch submit_training.slurm
```

## Monitoring Training

### TensorBoard

Monitor training progress in real-time:

```bash
source venv_codetr/bin/activate
tensorboard --logdir work_dirs/co_dino_5scale_r50_vinbigdata --port 6006
```

Then tunnel to your local machine:
```bash
# On your local machine
ssh -L 6006:localhost:6006 your_username@sol.asu.edu
```

Access at: http://localhost:6006

### Training Logs

Training logs are saved to:
- `work_dirs/co_dino_5scale_r50_vinbigdata/[timestamp].log`
- TensorBoard logs: `work_dirs/co_dino_5scale_r50_vinbigdata/tf_logs/`

## File Structure

```
/home/dpraja12/Dino/
├── Co-DETR/                          # Co-DETR repository
│   ├── projects/
│   │   └── configs/
│   │       ├── _base_/
│   │       │   └── datasets/
│   │       │       └── vinbigdata_detection.py  # Dataset config
│   │       └── co_dino/
│   │           └── co_dino_5scale_r50_vinbigdata.py  # Model config
│   └── tools/
│       ├── train.py
│       └── test.py
├── convert_vinbigdata_to_coco.py     # Dataset conversion script
├── setup_environment.sh              # Environment setup
├── train_vinbigdata.sh               # Training script
├── evaluate_model.sh                 # Evaluation script
├── work_dirs/                        # Training outputs (created during training)
└── venv_codetr/                      # Virtual environment (created by setup)
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:
1. Reduce batch size: Edit config, change `samples_per_gpu=2` or `1`
2. Reduce workers: Change `workers_per_gpu=2`
3. Reduce image size: Change `img_scale=(800, 800)` instead of 1024

### Slow Training

If training is too slow:
1. Ensure FP16 is enabled (it is by default)
2. Increase `workers_per_gpu` if CPU cores available
3. Check GPU utilization with `nvidia-smi`
4. Ensure data is on fast storage (not network drive)

### Import Errors

If you get import errors:
```bash
source venv_codetr/bin/activate
cd Co-DETR
pip install -e .
```

## Expected Results

After 36 epochs of training with ResNet-50:
- **mAP@0.5**: ~0.75-0.80
- **mAP@0.75**: ~0.50-0.60
- **mAP@[0.5:0.95]**: ~0.45-0.55

Note: These are estimates. Actual performance may vary based on:
- Data quality and distribution
- Hyperparameter tuning
- Training stability
- Random seed

## Citation

If you use this code, please cite Co-DETR:

```bibtex
@inproceedings{zong2023detrs,
  title={DETRs with Collaborative Hybrid Assignments Training},
  author={Zong, Zhuofan and Song, Guanglu and Liu, Yu},
  booktitle={ICCV},
  year={2023}
}
```

## Contact

For issues or questions, please check:
- Co-DETR GitHub: https://github.com/Sense-X/Co-DETR
- VinBigData Dataset: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

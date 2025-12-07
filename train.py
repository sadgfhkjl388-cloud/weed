#!/usr/bin/env python3
"""
Training Script for Cotton-Weed Detection Challenge

Simple, editable script for quick iteration - just modify the configuration
section below and run!

Usage:
    python train.py

Competition Rules:
    - YOLOv8n only (REQUIRED)
    - 640 input size (FIXED)
    - Hyperparameter tuning allowed
    - No ensembles

Learn more: See cotton_weed_starter_notebook.ipynb for explanations
"""

import torch
import tlc
from tlc_ultralytics import YOLO, Settings

# ============================================================================
# CONFIGURATION - Edit these values for your training run
# ============================================================================

# 3LC Table URLs (get these from Dashboard)
# Click your table -> Copy URL from browser or table info panel
TRAIN_TABLE_URL = "C:/Users/user/AppData/Local/3LC/3LC/projects/kaggle_cotton_weed_detection/datasets/cotton_weed_det3/tables/cotton_weed_det3-new_train"
VAL_TABLE_URL = "C:/Users/user/AppData/Local/3LC/3LC/projects/kaggle_cotton_weed_detection/datasets/cotton_weed_det3/tables/cotton_weed_det3-new_val"

# Example of table URLs 
# TRAIN_TABLE_URL = "C:/Users/rishi/AppData/Local/3LC/3LC/projects/kaggle_cotton_weed_detection/datasets/cotton_weed/tables/cotton_weed-train"
# VAL_TABLE_URL = "C:/Users/rishi/AppData/Local/3LC/3LC/projects/kaggle_cotton_weed_detection/datasets/cotton_weed/tables/cotton_weed-val"
    
    # Run configuration
PROJECT_NAME = "kaggle_cotton_weed_detection"  # 3LC project name
RUN_NAME = (
    "v1_quick_improvements"  # Quick improvements: epochs + augmentation + batch size
)
RUN_DESCRIPTION = "Quick improvements: 75 epochs + augmentation + batch 32"  # Describe this experiment

    # Training hyperparameters
EPOCHS = 300  # Number of training epochs
BATCH_SIZE = 32  # Batch size (reduce if GPU memory issues)
IMAGE_SIZE = 640  # Input image size (FIXED by competition)
DEVICE = 0  # GPU device (0 for first GPU, 'cpu' for CPU)
WORKERS = 4  # Number of dataloader workers

# Advanced hyperparameters (optional)
LR0 = 0.01  # Initial learning rate
PATIENCE = 30  # Early stopping patience (epochs without improvement)
SAVE_PERIOD = 1  # Save checkpoint every N epochs (1=every epoch, -1=only best/last)

# Optimizer settings
OPTIMIZER = 'SGD'  # Optimizer: 'SGD', 'Adam', 'AdamW' (default), 'NAdam', 'RAdam'
MOMENTUM = 0.937  # SGD momentum (0.8-0.99, only for SGD)
WEIGHT_DECAY = 0.0005  # L2 regularization weight decay

# Data augmentation (set to True to enable advanced augmentation)
USE_AUGMENTATION = True  # Enable mosaic, mixup, copy_paste

# ============================================================================
# TRAINING PIPELINE - No need to edit below this line
# ============================================================================


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("COTTON WEED DETECTION - TRAINING")
    print("=" * 70)

    # Check environment
    print("\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  3LC: {tlc.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Validate table URLs
    if "paste_your" in TRAIN_TABLE_URL or "paste_your" in VAL_TABLE_URL:
        print("\n !!! ERROR: Please set your table URLs in the configuration section!")
        print("\n How to get URLs:")
        print("   1. Open Dashboard: https://dashboard.3lc.ai")
        print("   2. Click on the tables tab")
        print("   3. Copy URL from table info panel to clipboard")
        print("   4. Paste URLs into TRAIN_TABLE_URL and VAL_TABLE_URL")
        return

    # Load tables
    print("\n" + "=" * 70)
    print("Loading Tables")
    print("=" * 70)

    print(f"\n Training table: {TRAIN_TABLE_URL}")
    train_table = tlc.Table.from_url(TRAIN_TABLE_URL)
    print(f"   OK - Loaded: {len(train_table)} samples")

    print(f"\n Validation table: {VAL_TABLE_URL}")
    val_table = tlc.Table.from_url(VAL_TABLE_URL)
    print(f"   OK - Loaded: {len(val_table)} samples")

    tables = {"train": train_table, "val": val_table}

    # Configure training
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"\n  Run: {RUN_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Device: {'GPU ' + str(DEVICE) if DEVICE != 'cpu' else 'CPU'}")
    print(f"  Learning rate: {LR0}")
    print(f"  Optimizer: {OPTIMIZER} (momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})")
    print(f"  Augmentation: {'Enabled' if USE_AUGMENTATION else 'Disabled'}")
    print(f"  Save period: {'Every epoch' if SAVE_PERIOD == 1 else f'Every {SAVE_PERIOD} epochs' if SAVE_PERIOD > 1 else 'Best/Last only'}")

    # Create 3LC Settings
    settings = Settings(
        project_name=PROJECT_NAME,
        run_name=RUN_NAME,
        run_description=RUN_DESCRIPTION,
        image_embeddings_dim=2,
    )

    # Load model
    print("\n Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print("   OK - Model loaded (3M parameters)")

    # Train
    print("\n" + "=" * 70)
    print("Training Started")
    print("=" * 70 + "\n")

    train_args = {
        "tables": tables,
        "name": RUN_NAME,
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "lr0": LR0,
        "patience": PATIENCE,
        "optimizer": OPTIMIZER,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "save_period": SAVE_PERIOD,
        "settings": settings,
        "val": True,
    }

    # Add augmentation if enabled
    if USE_AUGMENTATION:
        train_args.update(
            {
                "mosaic": 1.0,  # Mosaic augmentation
                "mixup": 0.1,  # Mixup augmentation
                "copy_paste": 0.2,  # Copy-paste augmentation
                # 新增以下進階增強:
                "hsv_h": 0.02,    # 色調變化（光線）
                "hsv_s": 0.9,      # 飽和度（天氣）
                "hsv_v": 0.6,      # 亮度
                "degrees": 15,     # 旋轉（相機角度）
                "translate": 0.15,  # 位置移動
                "scale": 0.7,      # 尺度變化
                "fliplr": 0.5,     # 水平翻轉
            }
            )

    model.train(**train_args)

    # Done!
    print("\n" + "=" * 70)
    print("OK - TRAINING COMPLETE!")
    print("=" * 70)

    if SAVE_PERIOD > 0:
        print(f"\n ✅ Weights saved:")
        print(f"    Best model: runs/detect/{RUN_NAME}/weights/best.pt")
        print(f"    Last epoch: runs/detect/{RUN_NAME}/weights/last.pt")
        print(f"    All epochs: runs/detect/{RUN_NAME}/weights/epoch*.pt")
    else:
        print(f"\n ✅ Weights saved: runs/detect/{RUN_NAME}/weights/best.pt")

    print("\n Next Steps:")
    print("   1. Check Dashboard: http://localhost:8000")
    print("   2. Analyze errors and edit data")
    print("   3. Generate predictions: python predict.py")
    print("   4. Retrain with edited data!")


if __name__ == "__main__":
    main()

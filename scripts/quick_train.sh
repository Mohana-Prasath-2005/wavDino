#!/bin/bash

# Quick training script for minimal experiments
# Uses CREMA-D only with baseline and temporal comparison

echo "=================================="
echo "Quick Training: CREMA-D Only"
echo "=================================="
echo ""
echo "This script will train:"
echo "1. Baseline (single frame)"
echo "2. Temporal 8-frame"
echo ""
echo "Estimated time: 6-8 hours with GPU"
echo "=================================="
echo ""

# Check if dataset exists
if [ ! -d "data/CREMA-D/AudioWAV" ] || [ ! -d "data/CREMA-D/VideoFlash" ]; then
    echo "ERROR: CREMA-D dataset not found!"
    echo "Please download CREMA-D first:"
    echo "  python scripts/setup_datasets.py --data_root ./data"
    echo ""
    echo "Then download from: https://github.com/CheyneyComputerScience/CREMA-D"
    exit 1
fi

echo "✓ CREMA-D dataset found"
echo ""

# Create output directory
OUTPUT_DIR="./quick_results"
mkdir -p $OUTPUT_DIR

# 1. Train baseline
echo "=================================="
echo "[1/2] Training Baseline (Single Frame)"
echo "=================================="
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir $OUTPUT_DIR/baseline \
    --device cuda \
    --num_workers 4

if [ $? -ne 0 ]; then
    echo "ERROR: Baseline training failed!"
    exit 1
fi

echo "✓ Baseline training complete"
echo ""

# 2. Train temporal 8-frame
echo "=================================="
echo "[2/2] Training Temporal (8 Frames)"
echo "=================================="
python train.py \
    --config configs/temporal_8frames.yaml \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir $OUTPUT_DIR/temporal_8frames \
    --device cuda \
    --num_workers 4

if [ $? -ne 0 ]; then
    echo "ERROR: Temporal training failed!"
    exit 1
fi

echo "✓ Temporal training complete"
echo ""

# Evaluate both models
echo "=================================="
echo "Evaluating Models"
echo "=================================="

echo "[1/2] Evaluating Baseline..."
python evaluate.py \
    --checkpoint $OUTPUT_DIR/baseline/checkpoint_best.pth \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir $OUTPUT_DIR/eval_baseline \
    --device cuda

echo "[2/2] Evaluating Temporal..."
python evaluate.py \
    --checkpoint $OUTPUT_DIR/temporal_8frames/checkpoint_best.pth \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir $OUTPUT_DIR/eval_temporal \
    --device cuda

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check these files:"
echo "  - $OUTPUT_DIR/eval_baseline/results_CREMA-D_test.json"
echo "  - $OUTPUT_DIR/eval_temporal/results_CREMA-D_test.json"
echo "  - $OUTPUT_DIR/eval_baseline/confusion_matrix_*.png"
echo "  - $OUTPUT_DIR/eval_temporal/confusion_matrix_*.png"
echo ""
echo "Use these results for your paper!"
echo "=================================="

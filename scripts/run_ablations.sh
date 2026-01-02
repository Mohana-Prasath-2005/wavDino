#!/bin/bash

# Run all ablation experiments for paper
# This script trains baseline and temporal models on all three datasets

echo "=================================="
echo "Running Ablation Experiments"
echo "=================================="

# Set dataset paths (MODIFY THESE)
CREMA_D_PATH="/path/to/CREMA-D"
RAVDESS_PATH="/path/to/RAVDESS"
AFEW_PATH="/path/to/AFEW"

OUTPUT_DIR="./ablation_results"
mkdir -p $OUTPUT_DIR

# ================================
# CREMA-D Experiments
# ================================

echo ""
echo "Starting CREMA-D experiments..."

# Baseline (Single Frame)
echo "  [1/4] Training baseline (single frame)..."
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset CREMA-D \
    --data_root $CREMA_D_PATH \
    --output_dir $OUTPUT_DIR/crema_d_baseline \
    --device cuda

# 8-Frame Temporal Attention
echo "  [2/4] Training 8-frame temporal attention..."
python train.py \
    --config configs/temporal_8frames.yaml \
    --dataset CREMA-D \
    --data_root $CREMA_D_PATH \
    --output_dir $OUTPUT_DIR/crema_d_8frames \
    --device cuda

# 16-Frame Temporal Attention
echo "  [3/4] Training 16-frame temporal attention..."
python train.py \
    --config configs/temporal_16frames.yaml \
    --dataset CREMA-D \
    --data_root $CREMA_D_PATH \
    --output_dir $OUTPUT_DIR/crema_d_16frames \
    --device cuda

# 8-Frame Temporal Transformer
echo "  [4/4] Training 8-frame temporal transformer..."
python train.py \
    --config configs/temporal_8frames_transformer.yaml \
    --dataset CREMA-D \
    --data_root $CREMA_D_PATH \
    --output_dir $OUTPUT_DIR/crema_d_8frames_transformer \
    --device cuda

# ================================
# RAVDESS Experiments
# ================================

echo ""
echo "Starting RAVDESS experiments..."

# Baseline
echo "  [1/4] Training baseline (single frame)..."
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset RAVDESS \
    --data_root $RAVDESS_PATH \
    --output_dir $OUTPUT_DIR/ravdess_baseline \
    --device cuda

# 8-Frame
echo "  [2/4] Training 8-frame temporal attention..."
python train.py \
    --config configs/temporal_8frames.yaml \
    --dataset RAVDESS \
    --data_root $RAVDESS_PATH \
    --output_dir $OUTPUT_DIR/ravdess_8frames \
    --device cuda

# 16-Frame
echo "  [3/4] Training 16-frame temporal attention..."
python train.py \
    --config configs/temporal_16frames.yaml \
    --dataset RAVDESS \
    --data_root $RAVDESS_PATH \
    --output_dir $OUTPUT_DIR/ravdess_16frames \
    --device cuda

# Transformer
echo "  [4/4] Training 8-frame temporal transformer..."
python train.py \
    --config configs/temporal_8frames_transformer.yaml \
    --dataset RAVDESS \
    --data_root $RAVDESS_PATH \
    --output_dir $OUTPUT_DIR/ravdess_8frames_transformer \
    --device cuda

# ================================
# AFEW Experiments
# ================================

echo ""
echo "Starting AFEW experiments..."

# Baseline
echo "  [1/4] Training baseline (single frame)..."
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset AFEW \
    --data_root $AFEW_PATH \
    --output_dir $OUTPUT_DIR/afew_baseline \
    --device cuda

# 8-Frame
echo "  [2/4] Training 8-frame temporal attention..."
python train.py \
    --config configs/temporal_8frames.yaml \
    --dataset AFEW \
    --data_root $AFEW_PATH \
    --output_dir $OUTPUT_DIR/afew_8frames \
    --device cuda

# 16-Frame
echo "  [3/4] Training 16-frame temporal attention..."
python train.py \
    --config configs/temporal_16frames.yaml \
    --dataset AFEW \
    --data_root $AFEW_PATH \
    --output_dir $OUTPUT_DIR/afew_16frames \
    --device cuda

# Transformer
echo "  [4/4] Training 8-frame temporal transformer..."
python train.py \
    --config configs/temporal_8frames_transformer.yaml \
    --dataset AFEW \
    --data_root $AFEW_PATH \
    --output_dir $OUTPUT_DIR/afew_8frames_transformer \
    --device cuda

echo ""
echo "=================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================="

# Generate comparison report
python scripts/generate_ablation_report.py --results_dir $OUTPUT_DIR

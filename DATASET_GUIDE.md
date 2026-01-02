# Quick Start Guide: Running with Online Datasets

This guide shows you how to download and use the datasets available online.

## 📊 Dataset Information

### 1. CREMA-D (Full Dataset - 100%)
- **Size**: ~8 GB
- **Samples**: 7,442 audio-video clips
- **Source**: https://github.com/CheyneyComputerScience/CREMA-D
- **Emotions**: Anger, Disgust, Fear, Happy, Neutral, Sad (6 classes)
- **Usage**: Full dataset for training and testing

### 2. RAVDESS (Full Dataset - 100%)
- **Size**: ~12 GB (with video)
- **Samples**: 1,440 audio-video clips (24 actors × 60 clips)
- **Source**: https://zenodo.org/record/1188976
- **Kaggle**: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
- **Emotions**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised (8 classes)
- **Usage**: Full dataset for training and testing

### 3. AFEW (10% Subset)
- **Size**: ~2 GB (10% subset)
- **Samples**: ~180 videos (from ~1,800 total)
- **Source**: https://cs.anu.edu.au/few/AFEW.html
- **Emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise (7 classes)
- **Usage**: 10% subset to reduce training time
- **Rationale**: AFEW is larger and takes longer to train; we focus on CREMA-D and RAVDESS

---

## 🚀 Step-by-Step Setup

### Step 1: Setup Dataset Directories

```bash
cd /home/mohan/Desktop/wavDino

# Run setup script to check status
python scripts/setup_datasets.py --data_root ./data --afew_percentage 10
```

This will create the directory structure and show download instructions.

### Step 2: Download CREMA-D

**Option A: Direct Download (Recommended)**
```bash
# Visit GitHub and download manually
# https://github.com/CheyneyComputerScience/CREMA-D

# Expected structure:
data/CREMA-D/
├── AudioWAV/
│   ├── 1001_DFA_ANG_XX.wav
│   └── ...
└── VideoFlash/
    ├── 1001_DFA_ANG_XX.flv
    └── ...
```

**Option B: Use Kaggle Dataset**
If available on Kaggle, you can use kaggle API:
```bash
# Install kaggle
pip install kaggle

# Download (if available)
kaggle datasets download -d <crema-d-dataset-name>
unzip <file>.zip -d data/CREMA-D/
```

### Step 3: Download RAVDESS

**Option A: Kaggle (Easiest)**
```bash
# RAVDESS is available on Kaggle
# https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio

# Download using kaggle CLI
pip install kaggle
kaggle datasets download -d uwrfkaggle/ravdess-emotional-speech-audio
unzip ravdess-emotional-speech-audio.zip -d data/RAVDESS/
```

**Option B: Zenodo (Official)**
```bash
# Visit: https://zenodo.org/record/1188976
# Download all Actor zip files
# Extract to data/RAVDESS/
```

Expected structure:
```
data/RAVDESS/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.mp4
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

### Step 4: Download AFEW (Optional, 10% will be used)

**Note**: AFEW requires registration and agreement to terms.

```bash
# Visit: https://cs.anu.edu.au/few/AFEW.html
# Register and download Train/Val splits
# Extract to data/AFEW/

# Expected structure:
data/AFEW/
├── Train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
└── Val/
    └── (same emotions)
```

**The code will automatically use only 10% of AFEW** to save time!

### Step 5: Verify Setup

```bash
# Run verification
python scripts/setup_datasets.py --data_root ./data

# You should see:
# CREMA-D        : ✓ Ready
# RAVDESS        : ✓ Ready
# AFEW           : ✓ Ready (or skip if not available)
```

---

## 🎯 Training Commands

### Train on CREMA-D (Full Dataset)

```bash
# Baseline (single frame)
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir ./outputs/crema_baseline

# Temporal (8 frames) - RECOMMENDED
python train.py \
    --config configs/temporal_8frames.yaml \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir ./outputs/crema_temporal_8frames
```

### Train on RAVDESS (Full Dataset)

```bash
# Baseline
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset RAVDESS \
    --data_root ./data/RAVDESS \
    --output_dir ./outputs/ravdess_baseline

# Temporal (8 frames)
python train.py \
    --config configs/temporal_8frames.yaml \
    --dataset RAVDESS \
    --data_root ./data/RAVDESS \
    --output_dir ./outputs/ravdess_temporal_8frames
```

### Train on AFEW (10% Subset)

```bash
# Baseline (10% of data)
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset AFEW \
    --data_root ./data/AFEW \
    --output_dir ./outputs/afew_baseline \
    --afew_percentage 10

# Temporal (16 frames recommended for AFEW, 10% of data)
python train.py \
    --config configs/temporal_16frames.yaml \
    --dataset AFEW \
    --data_root ./data/AFEW \
    --output_dir ./outputs/afew_temporal_16frames \
    --afew_percentage 10
```

**Note**: `--afew_percentage 10` is the default and automatically limits AFEW to 10%!

---

## 📈 Training Order (Recommended)

1. **Start with CREMA-D** (medium size, good performance)
   - Train baseline: ~2-3 hours
   - Train temporal 8-frame: ~3-4 hours

2. **Then RAVDESS** (best performance, clean data)
   - Train baseline: ~1-2 hours
   - Train temporal 8-frame: ~2-3 hours

3. **Finally AFEW (10%)** (challenging, in-the-wild)
   - Train baseline: ~1-2 hours (10% subset)
   - Train temporal 16-frame: ~2-3 hours (10% subset)

**Total time**: ~15-20 hours for all experiments (with GPU)

---

## 🎯 Minimal Training (To Save Time)

If you have limited time, focus on CREMA-D only:

```bash
# 1. Baseline on CREMA-D
python train.py \
    --config configs/baseline_single_frame.yaml \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir ./outputs/crema_baseline \
    --device cuda

# 2. Temporal 8-frame on CREMA-D
python train.py \
    --config configs/temporal_8frames.yaml \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir ./outputs/crema_temporal \
    --device cuda

# 3. Evaluate both
python evaluate.py \
    --checkpoint ./outputs/crema_baseline/checkpoint_best.pth \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir ./results/crema_baseline

python evaluate.py \
    --checkpoint ./outputs/crema_temporal/checkpoint_best.pth \
    --dataset CREMA-D \
    --data_root ./data/CREMA-D \
    --output_dir ./results/crema_temporal
```

This gives you:
- ✅ Baseline vs Temporal comparison
- ✅ Ablation results for paper
- ✅ Confusion matrices and metrics
- ✅ Complete in ~6-8 hours

---

## 📊 Expected File Sizes

```
data/
├── CREMA-D/          ~8 GB (100% - all files used)
├── RAVDESS/          ~12 GB (100% - all files used)
└── AFEW/             ~20 GB (only 10% used in training = ~2 GB effective)

Total Download: ~40 GB
Actual Usage: ~22 GB (10% of AFEW)
```

---

## 🔧 Troubleshooting

### Problem: "Dataset not found"
```bash
# Check directory structure
ls -R data/CREMA-D/
ls -R data/RAVDESS/

# Run setup script
python scripts/setup_datasets.py --data_root ./data
```

### Problem: "CUDA out of memory"
```bash
# Reduce batch size in config files
# Edit configs/*.yaml:
batch_size: 4  # Instead of 8

# Or use CPU (slower)
python train.py ... --device cpu
```

### Problem: "AFEW too large"
Don't worry! The code automatically uses only 10% of AFEW:
```bash
# This is already set by default
--afew_percentage 10
```

To use even less:
```bash
--afew_percentage 5  # Use only 5%
```

---

## ✅ Summary

**What You're Using:**
- ✅ CREMA-D: 100% (~7,400 samples)
- ✅ RAVDESS: 100% (~1,400 samples)
- ✅ AFEW: 10% (~180 samples)

**Why This Works:**
1. Focus on CREMA-D and RAVDESS (better quality, smaller size)
2. AFEW provides "in-the-wild" validation but is huge
3. 10% of AFEW is enough to show generalization
4. Total training time reduced by 70%

**Next Steps:**
1. Download datasets following instructions above
2. Run `python scripts/setup_datasets.py` to verify
3. Start with CREMA-D training
4. Generate results for paper

Good luck! 🚀

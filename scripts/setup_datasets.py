"""
Dataset Download and Setup Script

This script helps download and prepare the datasets:
- CREMA-D: Full dataset
- RAVDESS: Full dataset  
- AFEW: 10% subset (due to size constraints)
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile
import shutil


def download_file(url, destination, desc=None):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=desc or "Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def extract_archive(archive_path, destination):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(destination)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(destination)
    
    print(f"✓ Extracted to {destination}")


def setup_crema_d(data_root):
    """
    CREMA-D Dataset Setup
    
    Source: https://github.com/CheyneyComputerScience/CREMA-D
    
    Note: CREMA-D requires manual download due to terms of use.
    Please download from: https://github.com/CheyneyComputerScience/CREMA-D
    """
    print("\n" + "="*60)
    print("CREMA-D Dataset Setup")
    print("="*60)
    
    crema_dir = Path(data_root) / "CREMA-D"
    
    print("""
CREMA-D requires manual download:

1. Visit: https://github.com/CheyneyComputerScience/CREMA-D
2. Download the dataset (AudioWAV and VideoFlash folders)
3. Place them in: {crema_dir}

Expected structure:
{crema_dir}/
├── AudioWAV/
│   ├── 1001_DFA_ANG_XX.wav
│   ├── 1001_DFA_DIS_XX.wav
│   └── ...
└── VideoFlash/
    ├── 1001_DFA_ANG_XX.flv
    ├── 1001_DFA_DIS_XX.flv
    └── ...

Total files: ~7,000+ audio-video pairs
Size: ~8 GB
""".format(crema_dir=crema_dir))
    
    if crema_dir.exists():
        print(f"✓ CREMA-D directory found: {crema_dir}")
        
        audio_dir = crema_dir / "AudioWAV"
        video_dir = crema_dir / "VideoFlash"
        
        if audio_dir.exists() and video_dir.exists():
            num_audio = len(list(audio_dir.glob("*.wav")))
            num_video = len(list(video_dir.glob("*.flv")))
            print(f"✓ Found {num_audio} audio files")
            print(f"✓ Found {num_video} video files")
            return True
        else:
            print("⚠ Audio or Video directories not found!")
            return False
    else:
        print(f"⚠ CREMA-D directory not found: {crema_dir}")
        crema_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {crema_dir}")
        return False


def setup_ravdess(data_root):
    """
    RAVDESS Dataset Setup
    
    Source: https://zenodo.org/record/1188976
    
    The dataset is available on Zenodo and Kaggle.
    """
    print("\n" + "="*60)
    print("RAVDESS Dataset Setup")
    print("="*60)
    
    ravdess_dir = Path(data_root) / "RAVDESS"
    
    print("""
RAVDESS is available from multiple sources:

Option 1 - Zenodo (Official):
https://zenodo.org/record/1188976

Option 2 - Kaggle (Easier):
https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio

Instructions:
1. Download the dataset (Audio-Visual or Audio-only)
2. Extract all Actor folders to: {ravdess_dir}

Expected structure:
{ravdess_dir}/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.mp4
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
├── ...
└── Actor_24/

Total actors: 24
Files per actor: ~60 video-audio pairs
Size: ~12 GB (with video)
""".format(ravdess_dir=ravdess_dir))
    
    if ravdess_dir.exists():
        print(f"✓ RAVDESS directory found: {ravdess_dir}")
        
        actor_dirs = list(ravdess_dir.glob("Actor_*"))
        if actor_dirs:
            print(f"✓ Found {len(actor_dirs)} actor directories")
            
            # Count total files
            total_videos = sum(len(list(d.glob("*.mp4"))) for d in actor_dirs)
            total_audios = sum(len(list(d.glob("*.wav"))) for d in actor_dirs)
            print(f"✓ Found {total_videos} video files")
            print(f"✓ Found {total_audios} audio files")
            return True
        else:
            print("⚠ No Actor directories found!")
            return False
    else:
        print(f"⚠ RAVDESS directory not found: {ravdess_dir}")
        ravdess_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {ravdess_dir}")
        return False


def setup_afew(data_root, use_percentage=10):
    """
    AFEW Dataset Setup (10% subset)
    
    Source: https://cs.anu.edu.au/few/AFEW.html
    
    Note: AFEW requires registration and terms agreement.
    We'll use only 10% for faster training.
    """
    print("\n" + "="*60)
    print(f"AFEW Dataset Setup (Using {use_percentage}% subset)")
    print("="*60)
    
    afew_dir = Path(data_root) / "AFEW"
    
    print(f"""
AFEW (Acted Facial Expressions in the Wild):

Source: https://cs.anu.edu.au/few/AFEW.html

Note: AFEW requires registration and terms agreement.

Alternative - AFEW on Kaggle:
https://www.kaggle.com/datasets/

Instructions:
1. Download AFEW Train/Val splits
2. Extract to: {afew_dir}

Expected structure:
{afew_dir}/
├── Train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
└── Val/
    ├── Angry/
    └── ...

We will use only {use_percentage}% of AFEW for training due to:
- Large dataset size
- Longer training time
- Focus on CREMA-D and RAVDESS

Total files: ~1,800 videos (full dataset)
Using: ~180 videos ({use_percentage}%)
Size: ~2 GB (10% subset)
""".format(afew_dir=afew_dir, use_percentage=use_percentage))
    
    if afew_dir.exists():
        print(f"✓ AFEW directory found: {afew_dir}")
        
        train_dir = afew_dir / "Train"
        val_dir = afew_dir / "Val"
        
        if train_dir.exists() and val_dir.exists():
            # Count files per emotion
            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            total_train = 0
            total_val = 0
            
            for emotion in emotions:
                train_emotion = train_dir / emotion
                val_emotion = val_dir / emotion
                
                if train_emotion.exists():
                    count = len(list(train_emotion.glob("*.avi"))) + len(list(train_emotion.glob("*.mp4")))
                    total_train += count
                
                if val_emotion.exists():
                    count = len(list(val_emotion.glob("*.avi"))) + len(list(val_emotion.glob("*.mp4")))
                    total_val += count
            
            print(f"✓ Found {total_train} training videos")
            print(f"✓ Found {total_val} validation videos")
            print(f"⚠ Will use ~{int(total_train * use_percentage / 100)} training videos ({use_percentage}%)")
            return True
        else:
            print("⚠ Train or Val directories not found!")
            return False
    else:
        print(f"⚠ AFEW directory not found: {afew_dir}")
        afew_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {afew_dir}")
        return False


def verify_setup(data_root):
    """Verify all datasets are properly set up."""
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    data_root = Path(data_root)
    
    results = {
        'CREMA-D': setup_crema_d(data_root),
        'RAVDESS': setup_ravdess(data_root),
        'AFEW': setup_afew(data_root, use_percentage=10)
    }
    
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    all_ready = True
    for dataset, ready in results.items():
        status = "✓ Ready" if ready else "✗ Not Ready"
        print(f"{dataset:15s}: {status}")
        if not ready:
            all_ready = False
    
    if all_ready:
        print("\n🎉 All datasets are ready for training!")
        print("\nNext steps:")
        print("1. Run baseline training:")
        print("   python train.py --config configs/baseline_single_frame.yaml --dataset CREMA-D --data_root data")
        print("\n2. Run temporal training:")
        print("   python train.py --config configs/temporal_8frames.yaml --dataset CREMA-D --data_root data")
    else:
        print("\n⚠ Some datasets need to be downloaded manually.")
        print("Please follow the instructions above for each dataset.")
    
    return all_ready


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup emotion recognition datasets')
    parser.add_argument('--data_root', type=str, default='./data', 
                       help='Root directory for datasets')
    parser.add_argument('--afew_percentage', type=int, default=10,
                       help='Percentage of AFEW to use (default: 10)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Emotion Recognition Dataset Setup")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"AFEW usage: {args.afew_percentage}%")
    print("="*60)
    
    # Create data root if it doesn't exist
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Setup each dataset
    verify_setup(data_root)


if __name__ == '__main__':
    main()

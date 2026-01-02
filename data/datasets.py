"""
Dataset classes for CREMA-D, RAVDESS, and AFEW.

Each dataset handles:
- Loading audio and video files
- Extracting face frames using MTCNN
- Multi-frame sampling for temporal modeling
- Data preprocessing and augmentation
"""

import torch
from torch.utils.data import Dataset
import torchaudio
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os
from PIL import Image
from facenet_pytorch import MTCNN

from .transforms import get_video_transform, get_audio_transform, get_frame_sampler


# ============================================================================
# Base Dataset Class
# ============================================================================

class MultimodalEmotionDataset(Dataset):
    """
    Base class for multimodal emotion datasets.
    
    Handles common functionality:
    - Face detection and extraction
    - Multi-frame sampling
    - Audio-visual synchronization
    """
    
    def __init__(
        self,
        root_dir: str,
        num_frames: int = 8,
        img_size: int = 224,
        sampling_method: str = 'uniform',
        augment: bool = False,
        use_single_frame: bool = False  # For baseline ablation
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment
        self.use_single_frame = use_single_frame
        
        # Initialize transforms
        self.video_transform = get_video_transform(img_size=img_size, augment=augment)
        self.audio_transform = get_audio_transform(augment=augment)
        self.frame_sampler = get_frame_sampler(method=sampling_method, num_frames=num_frames)
        
        # Initialize face detector
        self.face_detector = MTCNN(
            image_size=img_size,
            margin=20,
            keep_all=False,
            post_process=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Emotion labels (override in subclasses)
        self.emotion_labels = []
        self.label_to_idx = {}
        
        # Data samples (to be filled by subclasses)
        self.samples = []
    
    def extract_faces_from_video(self, video_path: str) -> List[Image.Image]:
        """
        Extract face frames from video using MTCNN.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of PIL images containing detected faces
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frame indices
        if self.use_single_frame:
            # For baseline: just get middle frame
            frame_indices = [total_frames // 2]
        else:
            # For temporal: sample multiple frames
            frame_indices = self.frame_sampler(total_frames)
        
        faces = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            try:
                face = self.face_detector(frame_rgb)
                if face is not None:
                    # Convert to PIL Image
                    face_np = face.permute(1, 2, 0).numpy()
                    face_np = ((face_np + 1) * 127.5).astype(np.uint8)
                    face_pil = Image.fromarray(face_np)
                    faces.append(face_pil)
                else:
                    # If no face detected, use whole frame resized
                    face_pil = Image.fromarray(frame_rgb)
                    faces.append(face_pil)
            except Exception as e:
                # Fallback: use whole frame
                face_pil = Image.fromarray(frame_rgb)
                faces.append(face_pil)
        
        cap.release()
        
        # Handle case where no faces were extracted
        if len(faces) == 0:
            # Return blank frames
            blank = Image.new('RGB', (self.img_size, self.img_size), color=(128, 128, 128))
            faces = [blank] * (1 if self.use_single_frame else self.num_frames)
        
        # Pad if needed
        while len(faces) < (1 if self.use_single_frame else self.num_frames):
            faces.append(faces[-1])
        
        return faces
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Audio waveform tensor and sample rate
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform, sample_rate = self.audio_transform(waveform, sample_rate)
        return waveform, sample_rate
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


# ============================================================================
# CREMA-D Dataset
# ============================================================================

class CREMAD_Dataset(MultimodalEmotionDataset):
    """
    CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
    
    Emotions: Anger, Disgust, Fear, Happy, Neutral, Sad
    Format: <ActorID>_<Sentence>_<Emotion>_<Level>.mp4/.wav
    """
    
    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        super().__init__(root_dir, **kwargs)
        
        # Define emotion labels
        self.emotion_labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.emotion_labels)}
        
        # Load samples
        self._load_samples(split)
    
    def _load_samples(self, split: str):
        """Load CREMA-D samples."""
        video_dir = self.root_dir / 'VideoFlash'
        audio_dir = self.root_dir / 'AudioWAV'
        
        # Get all video files
        video_files = list(video_dir.glob('*.flv'))
        
        # Simple train/val/test split (can be improved with cross-validation)
        np.random.seed(42)
        np.random.shuffle(video_files)
        
        n_train = int(0.7 * len(video_files))
        n_val = int(0.15 * len(video_files))
        
        if split == 'train':
            video_files = video_files[:n_train]
        elif split == 'val':
            video_files = video_files[n_train:n_train + n_val]
        else:  # test
            video_files = video_files[n_train + n_val:]
        
        # Parse samples
        for video_file in video_files:
            # Parse filename: ActorID_Sentence_Emotion_Level.flv
            parts = video_file.stem.split('_')
            if len(parts) >= 3:
                emotion = parts[2]
                if emotion in self.label_to_idx:
                    audio_file = audio_dir / f"{video_file.stem}.wav"
                    if audio_file.exists():
                        self.samples.append({
                            'video_path': str(video_file),
                            'audio_path': str(audio_file),
                            'emotion': emotion,
                            'label': self.label_to_idx[emotion]
                        })
        
        print(f"CREMA-D {split}: {len(self.samples)} samples")
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video frames
        faces = self.extract_faces_from_video(sample['video_path'])
        
        # Load audio
        audio, _ = self.load_audio(sample['audio_path'])
        
        # Apply transforms
        if self.use_single_frame:
            video_tensor = self.video_transform(faces)[0]  # Single frame
        else:
            video_tensor = self.video_transform(faces)  # Multi-frame
        
        label = sample['label']
        
        return {
            'audio': audio,
            'video': video_tensor,
            'label': label,
            'emotion': sample['emotion']
        }


# ============================================================================
# RAVDESS Dataset
# ============================================================================

class RAVDESS_Dataset(MultimodalEmotionDataset):
    """
    RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
    
    Emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
    Format: 03-01-<Emotion>-<Intensity>-<Statement>-<Repetition>-<Actor>.mp4
    """
    
    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        super().__init__(root_dir, **kwargs)
        
        # Define emotion labels (01=neutral, 02=calm, ..., 08=surprised)
        self.emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        self.emotion_labels = list(self.emotion_map.values())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.emotion_labels)}
        
        # Load samples
        self._load_samples(split)
    
    def _load_samples(self, split: str):
        """Load RAVDESS samples."""
        # RAVDESS structure: Actor_XX/
        actor_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('Actor_')])
        
        # Split by actors (leave-one-out or simple split)
        np.random.seed(42)
        np.random.shuffle(actor_dirs)
        
        n_train = int(0.7 * len(actor_dirs))
        n_val = int(0.15 * len(actor_dirs))
        
        if split == 'train':
            actor_dirs = actor_dirs[:n_train]
        elif split == 'val':
            actor_dirs = actor_dirs[n_train:n_train + n_val]
        else:  # test
            actor_dirs = actor_dirs[n_train + n_val:]
        
        # Parse samples
        for actor_dir in actor_dirs:
            video_files = list(actor_dir.glob('*.mp4'))
            
            for video_file in video_files:
                # Parse filename
                parts = video_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in self.emotion_map:
                        emotion = self.emotion_map[emotion_code]
                        
                        # Audio file has same name but .wav
                        audio_file = video_file.with_suffix('.wav')
                        
                        if audio_file.exists():
                            self.samples.append({
                                'video_path': str(video_file),
                                'audio_path': str(audio_file),
                                'emotion': emotion,
                                'label': self.label_to_idx[emotion]
                            })
        
        print(f"RAVDESS {split}: {len(self.samples)} samples")
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video frames
        faces = self.extract_faces_from_video(sample['video_path'])
        
        # Load audio
        audio, _ = self.load_audio(sample['audio_path'])
        
        # Apply transforms
        if self.use_single_frame:
            video_tensor = self.video_transform(faces)[0]
        else:
            video_tensor = self.video_transform(faces)
        
        label = sample['label']
        
        return {
            'audio': audio,
            'video': video_tensor,
            'label': label,
            'emotion': sample['emotion']
        }


# ============================================================================
# AFEW Dataset
# ============================================================================

class AFEW_Dataset(MultimodalEmotionDataset):
    """
    AFEW: Acted Facial Expressions in the Wild
    
    Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
    Structure: Train/Val/<Emotion>/<video_files>
    
    Note: Can use subset of data (e.g., 10%) to reduce training time
    """
    
    def __init__(self, root_dir: str, split: str = 'train', use_percentage: int = 100, **kwargs):
        self.use_percentage = use_percentage
        super().__init__(root_dir, **kwargs)
        
        # Define emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.emotion_labels)}
        
        # Load samples
        self._load_samples(split)
    
    def _load_samples(self, split: str):
        """Load AFEW samples (with optional percentage subset)."""
        # AFEW structure: Train/Val/<Emotion>/
        split_dir = self.root_dir / ('Train' if split == 'train' else 'Val')
        
        all_samples = []
        
        for emotion in self.emotion_labels:
            emotion_dir = split_dir / emotion
            if emotion_dir.exists():
                video_files = list(emotion_dir.glob('*.avi')) + list(emotion_dir.glob('*.mp4'))
                
                for video_file in video_files:
                    # For AFEW, audio is embedded in video
                    all_samples.append({
                        'video_path': str(video_file),
                        'audio_path': str(video_file),  # Extract from video
                        'emotion': emotion,
                        'label': self.label_to_idx[emotion]
                    })
        
        # Apply percentage sampling if specified
        if self.use_percentage < 100:
            import random
            random.seed(42)  # Reproducible subset
            subset_size = int(len(all_samples) * self.use_percentage / 100)
            all_samples = random.sample(all_samples, subset_size)
            print(f"AFEW {split}: Using {self.use_percentage}% = {len(all_samples)} samples (from {len(all_samples) * 100 // self.use_percentage} total)")
        else:
            print(f"AFEW {split}: {len(all_samples)} samples")
        
        self.samples = all_samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video frames
        faces = self.extract_faces_from_video(sample['video_path'])
        
        # Load audio (extract from video)
        try:
            audio, _ = self.load_audio(sample['audio_path'])
        except:
            # If audio extraction fails, use silence
            audio = torch.zeros(16000 * 3)  # 3 seconds of silence
        
        # Apply transforms
        if self.use_single_frame:
            video_tensor = self.video_transform(faces)[0]
        else:
            video_tensor = self.video_transform(faces)
        
        label = sample['label']
        
        return {
            'audio': audio,
            'video': video_tensor,
            'label': label,
            'emotion': sample['emotion']
        }


# ============================================================================
# Dataset Factory
# ============================================================================

def create_dataset(
    dataset_name: str,
    root_dir: str,
    split: str = 'train',
    num_frames: int = 8,
    augment: bool = False,
    use_single_frame: bool = False,
    afew_percentage: int = 100
) -> MultimodalEmotionDataset:
    """
    Factory function to create datasets.
    
    Args:
        dataset_name: 'CREMA-D', 'RAVDESS', or 'AFEW'
        root_dir: Root directory of dataset
        split: 'train', 'val', or 'test'
        num_frames: Number of frames to sample
        augment: Whether to apply data augmentation
        use_single_frame: Use single frame (baseline ablation)
        afew_percentage: Percentage of AFEW to use (default: 100)
    
    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.upper().replace('-', '')
    
    if dataset_name == 'CREMA' or dataset_name == 'CREMAD':
        return CREMAD_Dataset(
            root_dir=root_dir,
            split=split,
            num_frames=num_frames,
            augment=augment,
            use_single_frame=use_single_frame
        )
    elif dataset_name == 'RAVDESS':
        return RAVDESS_Dataset(
            root_dir=root_dir,
            split=split,
            num_frames=num_frames,
            augment=augment,
            use_single_frame=use_single_frame
        )
    elif dataset_name == 'AFEW':
        return AFEW_Dataset(
            root_dir=root_dir,
            split=split,
            num_frames=num_frames,
            augment=augment,
            use_single_frame=use_single_frame,
            use_percentage=afew_percentage  # Use specified percentage
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch):
    """
    Custom collate function for handling variable-length audio.
    """
    # Find max audio length in batch
    max_audio_len = max([item['audio'].shape[0] for item in batch])
    
    # Pad audio to max length
    audios = []
    for item in batch:
        audio = item['audio']
        if audio.shape[0] < max_audio_len:
            padding = max_audio_len - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        audios.append(audio)
    
    # Stack tensors
    return {
        'audio': torch.stack(audios),
        'video': torch.stack([item['video'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch]),
        'emotion': [item['emotion'] for item in batch]
    }


if __name__ == "__main__":
    print("Dataset module loaded successfully!")
    print("Note: Actual testing requires dataset files to be present.")

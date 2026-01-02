"""
Data augmentation and transformations for audio and video.
"""

import torch
import torchaudio
import torchvision.transforms as T
import numpy as np
import random
from typing import Tuple


# ============================================================================
# Video/Image Transforms
# ============================================================================

class VideoTransform:
    """
    Video transformations for face frames.
    
    Standard preprocessing:
    - Resize to 224x224
    - Normalize with ImageNet stats (required by DINOv2)
    """
    
    def __init__(self, img_size: int = 224, augment: bool = False):
        self.img_size = img_size
        self.augment = augment
        
        # Base transform (always applied)
        base_transforms = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Augmentation transforms (training only)
        if augment:
            aug_transforms = [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomRotation(degrees=10),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.transform = T.Compose(aug_transforms)
        else:
            self.transform = T.Compose(base_transforms)
    
    def __call__(self, frames):
        """
        Apply transform to all frames.
        
        Args:
            frames: List of PIL images or numpy arrays
        
        Returns:
            Tensor of shape (num_frames, 3, H, W)
        """
        transformed_frames = [self.transform(frame) for frame in frames]
        return torch.stack(transformed_frames)


# ============================================================================
# Audio Transforms
# ============================================================================

class AudioTransform:
    """
    Audio transformations and augmentations.
    
    Standard preprocessing:
    - Resample to 16kHz (required by Wav2Vec)
    - Normalize amplitude
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        augment: bool = False,
        noise_prob: float = 0.3,
        noise_level: float = 0.005
    ):
        self.target_sample_rate = target_sample_rate
        self.augment = augment
        self.noise_prob = noise_prob
        self.noise_level = noise_level
    
    def __call__(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply audio transformations.
        
        Args:
            waveform: (channels, samples) audio tensor
            sample_rate: Original sample rate
        
        Returns:
            Transformed waveform and target sample rate
        """
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Augmentation (training only)
        if self.augment:
            # Add Gaussian noise
            if random.random() < self.noise_prob:
                noise = torch.randn_like(waveform) * self.noise_level
                waveform = waveform + noise
            
            # Random amplitude scaling
            if random.random() < 0.3:
                scale = random.uniform(0.8, 1.2)
                waveform = waveform * scale
        
        # Clip to [-1, 1]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform.squeeze(0), self.target_sample_rate  # Return 1D tensor


# ============================================================================
# Frame Sampling Strategies
# ============================================================================

class UniformFrameSampler:
    """
    Uniformly sample N frames from a video clip.
    
    This is the RECOMMENDED sampling strategy.
    """
    
    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
    
    def __call__(self, total_frames: int) -> list:
        """
        Get frame indices to sample.
        
        Args:
            total_frames: Total number of frames in video
        
        Returns:
            List of frame indices to extract
        """
        if total_frames <= self.num_frames:
            # If video is too short, repeat frames
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            return indices
        else:
            # Uniform sampling
            step = total_frames / self.num_frames
            indices = [int(i * step) for i in range(self.num_frames)]
            return indices


class RandomFrameSampler:
    """
    Randomly sample N frames from a video clip.
    
    Use this for data augmentation during training.
    """
    
    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
    
    def __call__(self, total_frames: int) -> list:
        """Get random frame indices."""
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(random.choice(indices))
            return sorted(indices)
        else:
            indices = sorted(random.sample(range(total_frames), self.num_frames))
            return indices


class DenseFrameSampler:
    """
    Sample frames from a dense temporal window.
    
    Useful for capturing short-term dynamics.
    """
    
    def __init__(self, num_frames: int = 8, window_size: int = None):
        self.num_frames = num_frames
        self.window_size = window_size or num_frames * 2
    
    def __call__(self, total_frames: int) -> list:
        """Sample frames from a dense window."""
        if total_frames <= self.window_size:
            return UniformFrameSampler(self.num_frames)(total_frames)
        
        # Random window start
        max_start = total_frames - self.window_size
        start = random.randint(0, max_start)
        
        # Uniform sampling within window
        step = self.window_size / self.num_frames
        indices = [start + int(i * step) for i in range(self.num_frames)]
        return indices


# ============================================================================
# Utility Functions
# ============================================================================

def get_video_transform(img_size: int = 224, augment: bool = False) -> VideoTransform:
    """Factory function for video transforms."""
    return VideoTransform(img_size=img_size, augment=augment)


def get_audio_transform(augment: bool = False) -> AudioTransform:
    """Factory function for audio transforms."""
    return AudioTransform(augment=augment)


def get_frame_sampler(method: str = 'uniform', num_frames: int = 8):
    """
    Factory function for frame samplers.
    
    Args:
        method: 'uniform', 'random', or 'dense'
        num_frames: Number of frames to sample
    
    Returns:
        Frame sampler instance
    """
    if method == 'uniform':
        return UniformFrameSampler(num_frames)
    elif method == 'random':
        return RandomFrameSampler(num_frames)
    elif method == 'dense':
        return DenseFrameSampler(num_frames)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("Testing Transforms...")
    
    # Test video transform
    print("\n=== Testing Video Transform ===")
    video_transform = get_video_transform(augment=True)
    dummy_frames = [torch.randn(224, 224, 3) for _ in range(8)]
    transformed = video_transform(dummy_frames)
    print(f"Input: {len(dummy_frames)} frames")
    print(f"Output shape: {transformed.shape}")
    
    # Test audio transform
    print("\n=== Testing Audio Transform ===")
    audio_transform = get_audio_transform(augment=True)
    dummy_audio = torch.randn(1, 48000)  # 1 channel, 48kHz
    transformed_audio, sr = audio_transform(dummy_audio, 48000)
    print(f"Input shape: {dummy_audio.shape}, SR: 48000")
    print(f"Output shape: {transformed_audio.shape}, SR: {sr}")
    
    # Test frame samplers
    print("\n=== Testing Frame Samplers ===")
    for method in ['uniform', 'random', 'dense']:
        sampler = get_frame_sampler(method=method, num_frames=8)
        indices = sampler(total_frames=100)
        print(f"{method.capitalize()} sampler: {indices}")
    
    print("\n✓ All tests passed!")

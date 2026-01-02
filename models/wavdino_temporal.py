"""
Complete Temporal wavDINO-Emotion Model

This module combines all components:
- Audio: Wav2Vec 2.0
- Visual: Temporal DINOv2 (multi-frame with temporal attention)
- Fusion: Multimodal transformer
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .audio_encoder import AudioEncoder
from .temporal_visual_encoder import TemporalVisualEncoder, SingleFrameVisualEncoder
from .multimodal_fusion import MultimodalFusionTransformer


class TemporalWavDINO(nn.Module):
    """
    Temporal wavDINO-Emotion: Enhanced multimodal emotion recognition.
    
    This is the MAIN MODEL for your enhanced paper.
    
    Architecture:
        Audio Branch:  Waveform → Wav2Vec 2.0 → A ∈ R^768
        Visual Branch: N Frames → DINOv2 → Temporal Attention → V ∈ R^1024
        Fusion:        [A, V] → Multimodal Transformer → Emotion Logits
    """
    
    def __init__(
        self,
        # Audio encoder config
        audio_model_name: str = 'facebook/wav2vec2-base-960h',
        freeze_wav2vec: bool = True,
        audio_pool_method: str = 'mean',
        
        # Visual encoder config
        visual_model_name: str = 'dinov2_vitl14',
        freeze_dinov2: bool = True,
        temporal_method: str = 'attention',  # 'attention' or 'transformer'
        num_frames: int = 8,  # 8 or 16 recommended
        
        # Fusion config
        fusion_dim: int = 512,
        fusion_method: str = 'transformer',
        num_fusion_heads: int = 8,
        num_fusion_layers: int = 2,
        
        # Classification config
        num_classes: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # Audio encoder (Wav2Vec 2.0)
        self.audio_encoder = AudioEncoder(
            model_name=audio_model_name,
            freeze_wav2vec=freeze_wav2vec,
            pool_method=audio_pool_method
        )
        audio_dim = 768  # Wav2Vec base output
        
        # Visual encoder (Temporal DINOv2)
        self.visual_encoder = TemporalVisualEncoder(
            dinov2_model_name=visual_model_name,
            temporal_method=temporal_method,
            freeze_dinov2=freeze_dinov2,
            dropout=dropout
        )
        visual_dim = 1024  # DINOv2 ViT-L output
        
        # Multimodal fusion
        self.fusion = MultimodalFusionTransformer(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            fusion_dim=fusion_dim,
            num_heads=num_fusion_heads,
            num_layers=num_fusion_layers,
            num_classes=num_classes,
            dropout=dropout,
            fusion_method=fusion_method
        )
        
    def forward(
        self, 
        audio: torch.Tensor,
        video_frames: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            audio: (batch_size, audio_length) - raw waveform at 16kHz
            video_frames: (batch_size, num_frames, 3, H, W) - face frames
            audio_mask: (batch_size, audio_length) - optional padding mask
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Extract audio features
        audio_features = self.audio_encoder(audio, attention_mask=audio_mask)
        
        # Extract visual features with temporal modeling
        visual_features = self.visual_encoder(video_frames)
        
        # Multimodal fusion and classification
        logits = self.fusion(audio_features, visual_features)
        
        return logits
    
    def get_num_params(self) -> Dict[str, Dict[str, int]]:
        """Get parameter counts for each component."""
        return {
            'audio_encoder': self.audio_encoder.get_num_params(),
            'visual_encoder': self.visual_encoder.get_num_params(),
            'fusion': self.fusion.get_num_params(),
            'total': {
                'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
            }
        }


class BaselineWavDINO(nn.Module):
    """
    Baseline wavDINO-Emotion with SINGLE-FRAME visual encoding.
    
    This is for ABLATION comparison - your original 7th semester paper.
    """
    
    def __init__(
        self,
        # Audio encoder config
        audio_model_name: str = 'facebook/wav2vec2-base-960h',
        freeze_wav2vec: bool = True,
        audio_pool_method: str = 'mean',
        
        # Visual encoder config
        visual_model_name: str = 'dinov2_vitl14',
        freeze_dinov2: bool = True,
        
        # Fusion config
        fusion_dim: int = 512,
        fusion_method: str = 'transformer',
        num_fusion_heads: int = 8,
        num_fusion_layers: int = 2,
        
        # Classification config
        num_classes: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Audio encoder (same as temporal version)
        self.audio_encoder = AudioEncoder(
            model_name=audio_model_name,
            freeze_wav2vec=freeze_wav2vec,
            pool_method=audio_pool_method
        )
        audio_dim = 768
        
        # Visual encoder (SINGLE FRAME - original paper)
        self.visual_encoder = SingleFrameVisualEncoder(
            dinov2_model_name=visual_model_name,
            freeze_dinov2=freeze_dinov2
        )
        visual_dim = 1024
        
        # Multimodal fusion (same as temporal version)
        self.fusion = MultimodalFusionTransformer(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            fusion_dim=fusion_dim,
            num_heads=num_fusion_heads,
            num_layers=num_fusion_layers,
            num_classes=num_classes,
            dropout=dropout,
            fusion_method=fusion_method
        )
        
    def forward(
        self, 
        audio: torch.Tensor,
        video_frame: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            audio: (batch_size, audio_length)
            video_frame: (batch_size, 3, H, W) - SINGLE frame
            audio_mask: (batch_size, audio_length)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Extract features
        audio_features = self.audio_encoder(audio, attention_mask=audio_mask)
        visual_features = self.visual_encoder(video_frame)
        
        # Fusion
        logits = self.fusion(audio_features, visual_features)
        
        return logits
    
    def get_num_params(self) -> Dict[str, Dict[str, int]]:
        """Get parameter counts for each component."""
        return {
            'audio_encoder': self.audio_encoder.get_num_params(),
            'visual_encoder': {'total': sum(p.numel() for p in self.visual_encoder.parameters()),
                               'trainable': sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)},
            'fusion': self.fusion.get_num_params(),
            'total': {
                'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
            }
        }


# ============================================================================
# Model Factory
# ============================================================================

def create_model(config: Dict) -> nn.Module:
    """
    Factory function to create models based on config.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        model: Instantiated model
    """
    model_type = config.get('model_type', 'temporal')
    
    if model_type == 'temporal':
        model = TemporalWavDINO(
            audio_model_name=config.get('audio_model', 'facebook/wav2vec2-base-960h'),
            freeze_wav2vec=config.get('freeze_wav2vec', True),
            visual_model_name=config.get('visual_model', 'dinov2_vitl14'),
            freeze_dinov2=config.get('freeze_dinov2', True),
            temporal_method=config.get('temporal_method', 'attention'),
            num_frames=config.get('num_frames', 8),
            fusion_dim=config.get('fusion_dim', 512),
            fusion_method=config.get('fusion_method', 'transformer'),
            num_classes=config.get('num_classes', 7),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'baseline':
        model = BaselineWavDINO(
            audio_model_name=config.get('audio_model', 'facebook/wav2vec2-base-960h'),
            freeze_wav2vec=config.get('freeze_wav2vec', True),
            visual_model_name=config.get('visual_model', 'dinov2_vitl14'),
            freeze_dinov2=config.get('freeze_dinov2', True),
            fusion_dim=config.get('fusion_dim', 512),
            fusion_method=config.get('fusion_method', 'transformer'),
            num_classes=config.get('num_classes', 7),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("Testing Complete Models...")
    
    # Test configuration
    batch_size = 2
    num_frames = 8
    audio_length = 16000 * 3  # 3 seconds
    img_size = 224
    
    # Create dummy inputs
    audio = torch.randn(batch_size, audio_length)
    video_frames = torch.randn(batch_size, num_frames, 3, img_size, img_size)
    single_frame = video_frames[:, 0, :, :, :]  # For baseline
    
    # Test temporal model
    print("\n=== Testing Temporal wavDINO (Enhanced) ===")
    temporal_model = TemporalWavDINO(
        num_frames=num_frames,
        temporal_method='attention',
        num_classes=7
    )
    logits_temporal = temporal_model(audio, video_frames)
    print(f"Audio shape: {audio.shape}")
    print(f"Video frames shape: {video_frames.shape}")
    print(f"Output logits shape: {logits_temporal.shape}")
    print(f"Parameters: {temporal_model.get_num_params()}")
    
    # Test baseline model
    print("\n=== Testing Baseline wavDINO (Original) ===")
    baseline_model = BaselineWavDINO(num_classes=7)
    logits_baseline = baseline_model(audio, single_frame)
    print(f"Audio shape: {audio.shape}")
    print(f"Video frame shape: {single_frame.shape}")
    print(f"Output logits shape: {logits_baseline.shape}")
    print(f"Parameters: {baseline_model.get_num_params()}")
    
    # Test model factory
    print("\n=== Testing Model Factory ===")
    config = {
        'model_type': 'temporal',
        'num_frames': 16,
        'temporal_method': 'transformer',
        'num_classes': 7
    }
    factory_model = create_model(config)
    print(f"Created model type: {type(factory_model).__name__}")
    
    print("\n✓ All tests passed!")

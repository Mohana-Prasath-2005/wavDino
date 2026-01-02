"""
Temporal Visual Encoder with Multi-Frame DINOv2 and Temporal Attention

This module implements the enhanced visual encoding that processes multiple frames
instead of a single static frame, capturing temporal dynamics of emotions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TemporalAttentionPooling(nn.Module):
    """
    Attention-weighted temporal pooling for frame aggregation.
    
    This is the RECOMMENDED approach (Option A1) - simple, fast, and effective.
    Computes learnable attention weights across frames and aggregates features.
    """
    
    def __init__(self, embed_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Attention projection
        self.attention_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )
        
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch_size, num_frames, embed_dim)
        
        Returns:
            aggregated_features: (batch_size, embed_dim)
        """
        # Compute attention scores
        attention_logits = self.attention_proj(frame_features)  # (B, N, 1)
        attention_weights = F.softmax(attention_logits, dim=1)  # (B, N, 1)
        
        # Weighted aggregation
        aggregated = torch.sum(attention_weights * frame_features, dim=1)  # (B, embed_dim)
        
        return aggregated


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for frame sequence modeling.
    
    This is the STRONGER approach (Option A2) - more parameters but better performance.
    Uses transformer encoder layers to model temporal relationships between frames.
    """
    
    def __init__(
        self, 
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # CLS token for aggregation (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch_size, num_frames, embed_dim)
        
        Returns:
            temporal_features: (batch_size, embed_dim)
        """
        batch_size = frame_features.size(0)
        
        # Add CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
            frame_features = torch.cat([cls_tokens, frame_features], dim=1)  # (B, N+1, embed_dim)
        
        # Add positional encoding
        frame_features = self.positional_encoding(frame_features)
        
        # Apply transformer
        transformed = self.transformer(frame_features)  # (B, N+1, embed_dim)
        
        # Aggregate: use CLS token or mean pooling
        if self.use_cls_token:
            output = transformed[:, 0, :]  # (B, embed_dim)
        else:
            output = transformed.mean(dim=1)  # (B, embed_dim)
        
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalVisualEncoder(nn.Module):
    """
    Complete Temporal Visual Encoder with DINOv2 + Temporal Modeling.
    
    This is the main module that replaces the single-frame visual encoder.
    
    Pipeline:
        1. Extract DINOv2 features for each frame independently
        2. Apply temporal modeling (attention pooling or transformer)
        3. Output single aggregated visual representation
    """
    
    def __init__(
        self,
        dinov2_model_name: str = 'dinov2_vitl14',
        temporal_method: str = 'attention',  # 'attention' or 'transformer'
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        freeze_dinov2: bool = True
    ):
        """
        Args:
            dinov2_model_name: DINOv2 model variant
            temporal_method: 'attention' (Option A1) or 'transformer' (Option A2)
            embed_dim: Feature dimension (DINOv2 output)
            num_heads: Number of attention heads (for transformer)
            num_layers: Number of transformer layers
            dropout: Dropout probability
            freeze_dinov2: Whether to freeze DINOv2 weights
        """
        super().__init__()
        self.temporal_method = temporal_method
        self.freeze_dinov2 = freeze_dinov2
        
        # Load DINOv2 model
        self.dinov2 = self._load_dinov2(dinov2_model_name)
        
        # Freeze DINOv2 if requested (recommended to reduce training time)
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
        
        # Temporal modeling module
        if temporal_method == 'attention':
            self.temporal_aggregator = TemporalAttentionPooling(
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif temporal_method == 'transformer':
            self.temporal_aggregator = TemporalTransformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown temporal method: {temporal_method}")
        
    def _load_dinov2(self, model_name: str):
        """Load pre-trained DINOv2 model."""
        try:
            # Try loading from torch hub
            dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
            return dinov2
        except Exception as e:
            print(f"Warning: Could not load DINOv2 from hub: {e}")
            print("Falling back to timm...")
            import timm
            # Map to timm model names
            timm_name = f'vit_large_patch14_dinov2.lvd142m' if 'vitl14' in model_name else 'vit_base_patch14_dinov2.lvd142m'
            dinov2 = timm.create_model(timm_name, pretrained=True, num_classes=0)
            return dinov2
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (batch_size, num_frames, 3, H, W) - typically H=W=224
        
        Returns:
            visual_features: (batch_size, embed_dim)
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # Reshape to process all frames together: (B*N, 3, H, W)
        frames_flat = frames.view(batch_size * num_frames, C, H, W)
        
        # Extract DINOv2 features for all frames
        if self.freeze_dinov2:
            with torch.no_grad():
                frame_features_flat = self.dinov2(frames_flat)  # (B*N, embed_dim)
        else:
            frame_features_flat = self.dinov2(frames_flat)  # (B*N, embed_dim)
        
        # Reshape back to separate frames: (B, N, embed_dim)
        embed_dim = frame_features_flat.size(-1)
        frame_features = frame_features_flat.view(batch_size, num_frames, embed_dim)
        
        # Apply temporal modeling
        temporal_features = self.temporal_aggregator(frame_features)  # (B, embed_dim)
        
        return temporal_features
    
    def get_num_params(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ============================================================================
# Helper function for single-frame baseline (for ablation comparison)
# ============================================================================

class SingleFrameVisualEncoder(nn.Module):
    """
    Baseline: Single-frame visual encoder (your original paper).
    
    This is kept for ablation comparison.
    """
    
    def __init__(
        self,
        dinov2_model_name: str = 'dinov2_vitl14',
        freeze_dinov2: bool = True
    ):
        super().__init__()
        self.freeze_dinov2 = freeze_dinov2
        
        # Load DINOv2
        try:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_model_name)
        except:
            import timm
            timm_name = f'vit_large_patch14_dinov2.lvd142m' if 'vitl14' in dinov2_model_name else 'vit_base_patch14_dinov2.lvd142m'
            self.dinov2 = timm.create_model(timm_name, pretrained=True, num_classes=0)
        
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
    
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame: (batch_size, 3, H, W)
        
        Returns:
            visual_features: (batch_size, embed_dim)
        """
        if self.freeze_dinov2:
            with torch.no_grad():
                features = self.dinov2(frame)
        else:
            features = self.dinov2(frame)
        
        return features


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("Testing Temporal Visual Encoder...")
    
    # Test configurations
    batch_size = 4
    num_frames = 8
    img_size = 224
    
    # Create dummy input
    frames = torch.randn(batch_size, num_frames, 3, img_size, img_size)
    
    # Test Option A1: Attention Pooling (RECOMMENDED)
    print("\n=== Testing Attention Pooling ===")
    encoder_att = TemporalVisualEncoder(
        temporal_method='attention',
        freeze_dinov2=True
    )
    output_att = encoder_att(frames)
    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {output_att.shape}")
    print(f"Parameters: {encoder_att.get_num_params()}")
    
    # Test Option A2: Temporal Transformer
    print("\n=== Testing Temporal Transformer ===")
    encoder_trans = TemporalVisualEncoder(
        temporal_method='transformer',
        freeze_dinov2=True
    )
    output_trans = encoder_trans(frames)
    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {output_trans.shape}")
    print(f"Parameters: {encoder_trans.get_num_params()}")
    
    # Test baseline
    print("\n=== Testing Single Frame Baseline ===")
    single_frame = frames[:, 0, :, :, :]  # Just first frame
    encoder_baseline = SingleFrameVisualEncoder(freeze_dinov2=True)
    output_baseline = encoder_baseline(single_frame)
    print(f"Input shape: {single_frame.shape}")
    print(f"Output shape: {output_baseline.shape}")
    
    print("\n✓ All tests passed!")

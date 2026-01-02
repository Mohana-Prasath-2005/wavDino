"""
Multimodal Fusion Model

Combines audio and visual features using a transformer-based fusion mechanism.
This is kept mostly the SAME as your original paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionTransformer(nn.Module):
    """
    Multimodal fusion using cross-attention transformer.
    
    Fuses audio (Wav2Vec 2.0) and visual (Temporal DINOv2) features
    for final emotion classification.
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        visual_dim: int = 1024,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.1,
        fusion_method: str = 'transformer'  # 'transformer', 'concat', or 'attention'
    ):
        """
        Args:
            audio_dim: Dimension of audio features (768 for Wav2Vec base)
            visual_dim: Dimension of visual features (1024 for DINOv2)
            fusion_dim: Dimension of fused representation
            num_heads: Number of attention heads
            num_layers: Number of fusion transformer layers
            num_classes: Number of emotion classes
            dropout: Dropout probability
            fusion_method: How to fuse modalities
        """
        super().__init__()
        self.fusion_method = fusion_method
        
        # Project audio and visual to same dimension
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        
        # Fusion mechanism
        if fusion_method == 'transformer':
            # Transformer-based fusion (original paper approach)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
        elif fusion_method == 'attention':
            # Cross-modal attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
        elif fusion_method == 'concat':
            # Simple concatenation (baseline)
            pass
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        if fusion_method == 'concat':
            classifier_input_dim = fusion_dim * 2
        else:
            classifier_input_dim = fusion_dim
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Modality tokens (for transformer fusion)
        self.audio_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.visual_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        
    def forward(
        self, 
        audio_features: torch.Tensor, 
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, audio_dim)
            visual_features: (batch_size, visual_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = audio_features.size(0)
        
        # Project to fusion dimension
        audio_proj = self.audio_proj(audio_features)  # (B, fusion_dim)
        visual_proj = self.visual_proj(visual_features)  # (B, fusion_dim)
        
        # Apply fusion
        if self.fusion_method == 'transformer':
            # Stack as sequence with modality tokens
            audio_seq = audio_proj.unsqueeze(1) + self.audio_token  # (B, 1, fusion_dim)
            visual_seq = visual_proj.unsqueeze(1) + self.visual_token  # (B, 1, fusion_dim)
            
            # Concatenate modalities
            multimodal_seq = torch.cat([audio_seq, visual_seq], dim=1)  # (B, 2, fusion_dim)
            
            # Apply transformer fusion
            fused = self.fusion_transformer(multimodal_seq)  # (B, 2, fusion_dim)
            
            # Aggregate (mean pooling)
            fused = fused.mean(dim=1)  # (B, fusion_dim)
            
        elif self.fusion_method == 'attention':
            # Cross-modal attention: visual attends to audio
            audio_seq = audio_proj.unsqueeze(1)  # (B, 1, fusion_dim)
            visual_seq = visual_proj.unsqueeze(1)  # (B, 1, fusion_dim)
            
            # Visual queries audio
            attended, _ = self.cross_attention(visual_seq, audio_seq, audio_seq)
            fused = (attended.squeeze(1) + visual_proj) / 2  # Residual connection
            
        elif self.fusion_method == 'concat':
            # Simple concatenation
            fused = torch.cat([audio_proj, visual_proj], dim=1)  # (B, fusion_dim * 2)
        
        # Classification
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits
    
    def get_num_params(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("Testing Multimodal Fusion...")
    
    # Test configuration
    batch_size = 4
    audio_dim = 768
    visual_dim = 1024
    num_classes = 7
    
    # Create dummy features
    audio_features = torch.randn(batch_size, audio_dim)
    visual_features = torch.randn(batch_size, visual_dim)
    
    # Test transformer fusion
    print("\n=== Testing Transformer Fusion ===")
    model_transformer = MultimodalFusionTransformer(
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        fusion_dim=512,
        num_classes=num_classes,
        fusion_method='transformer'
    )
    logits_transformer = model_transformer(audio_features, visual_features)
    print(f"Audio shape: {audio_features.shape}")
    print(f"Visual shape: {visual_features.shape}")
    print(f"Output logits shape: {logits_transformer.shape}")
    print(f"Parameters: {model_transformer.get_num_params()}")
    
    # Test attention fusion
    print("\n=== Testing Attention Fusion ===")
    model_attention = MultimodalFusionTransformer(
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        fusion_dim=512,
        num_classes=num_classes,
        fusion_method='attention'
    )
    logits_attention = model_attention(audio_features, visual_features)
    print(f"Output logits shape: {logits_attention.shape}")
    print(f"Parameters: {model_attention.get_num_params()}")
    
    # Test concat fusion
    print("\n=== Testing Concat Fusion ===")
    model_concat = MultimodalFusionTransformer(
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        fusion_dim=512,
        num_classes=num_classes,
        fusion_method='concat'
    )
    logits_concat = model_concat(audio_features, visual_features)
    print(f"Output logits shape: {logits_concat.shape}")
    print(f"Parameters: {model_concat.get_num_params()}")
    
    print("\n✓ All tests passed!")

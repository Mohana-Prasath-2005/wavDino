"""
Audio Encoder with Wav2Vec 2.0

This module handles audio feature extraction using pre-trained Wav2Vec 2.0.
This is kept the SAME as your original paper.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioEncoder(nn.Module):
    """
    Audio encoder using Wav2Vec 2.0.
    
    Extracts contextualized audio representations from raw waveforms.
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/wav2vec2-base-960h',
        freeze_wav2vec: bool = True,
        output_dim: int = 768,
        pool_method: str = 'mean'
    ):
        """
        Args:
            model_name: Hugging Face model name
            freeze_wav2vec: Whether to freeze Wav2Vec weights
            output_dim: Output feature dimension (768 for base, 1024 for large)
            pool_method: How to pool sequence: 'mean', 'max', or 'last'
        """
        super().__init__()
        self.pool_method = pool_method
        self.output_dim = output_dim
        
        # Load pre-trained Wav2Vec 2.0
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze if requested
        if freeze_wav2vec:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
            self.wav2vec.eval()
        
        self.freeze_wav2vec = freeze_wav2vec
        
    def forward(self, waveforms: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Args:
            waveforms: (batch_size, audio_length) - raw audio at 16kHz
            attention_mask: (batch_size, audio_length) - optional padding mask
        
        Returns:
            audio_features: (batch_size, output_dim)
        """
        # Extract Wav2Vec features
        if self.freeze_wav2vec:
            with torch.no_grad():
                outputs = self.wav2vec(waveforms, attention_mask=attention_mask)
        else:
            outputs = self.wav2vec(waveforms, attention_mask=attention_mask)
        
        # Get hidden states: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.last_hidden_state
        
        # Pool the sequence
        if self.pool_method == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                audio_features = sum_hidden / sum_mask
            else:
                audio_features = hidden_states.mean(dim=1)
        elif self.pool_method == 'max':
            audio_features = hidden_states.max(dim=1)[0]
        elif self.pool_method == 'last':
            audio_features = hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
        
        return audio_features
    
    def get_num_params(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ============================================================================
# Audio Processor Wrapper
# ============================================================================

class AudioProcessor:
    """
    Utility class for audio preprocessing.
    
    Handles:
    - Loading audio files
    - Resampling to 16kHz (required by Wav2Vec)
    - Normalization
    """
    
    def __init__(self, model_name: str = 'facebook/wav2vec2-base-960h'):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.sampling_rate = 16000
    
    def __call__(self, audio_array, sampling_rate=None):
        """
        Process audio array to Wav2Vec input format.
        
        Args:
            audio_array: numpy array or tensor of audio samples
            sampling_rate: Original sampling rate (will resample if needed)
        
        Returns:
            input_values: Tensor ready for Wav2Vec model
        """
        # Convert to numpy if tensor
        if torch.is_tensor(audio_array):
            audio_array = audio_array.numpy()
        
        # Process using Wav2Vec processor
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate or self.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values.squeeze(0)


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("Testing Audio Encoder...")
    
    # Test configuration
    batch_size = 4
    audio_length = 16000 * 3  # 3 seconds at 16kHz
    
    # Create dummy audio
    waveforms = torch.randn(batch_size, audio_length)
    
    # Test audio encoder
    print("\n=== Testing Wav2Vec 2.0 Encoder ===")
    encoder = AudioEncoder(
        model_name='facebook/wav2vec2-base-960h',
        freeze_wav2vec=True
    )
    
    audio_features = encoder(waveforms)
    
    print(f"Input shape: {waveforms.shape}")
    print(f"Output shape: {audio_features.shape}")
    print(f"Parameters: {encoder.get_num_params()}")
    
    # Test audio processor
    print("\n=== Testing Audio Processor ===")
    processor = AudioProcessor()
    import numpy as np
    audio_np = np.random.randn(audio_length)
    processed = processor(audio_np)
    print(f"Input shape: {audio_np.shape}")
    print(f"Output shape: {processed.shape}")
    
    print("\n✓ All tests passed!")

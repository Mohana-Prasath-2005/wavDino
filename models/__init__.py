"""Models package for Temporal wavDINO-Emotion."""

from .audio_encoder import AudioEncoder, AudioProcessor
from .temporal_visual_encoder import (
    TemporalVisualEncoder,
    SingleFrameVisualEncoder,
    TemporalAttentionPooling,
    TemporalTransformer
)
from .multimodal_fusion import MultimodalFusionTransformer
from .wavdino_temporal import (
    TemporalWavDINO,
    BaselineWavDINO,
    create_model
)

__all__ = [
    'AudioEncoder',
    'AudioProcessor',
    'TemporalVisualEncoder',
    'SingleFrameVisualEncoder',
    'TemporalAttentionPooling',
    'TemporalTransformer',
    'MultimodalFusionTransformer',
    'TemporalWavDINO',
    'BaselineWavDINO',
    'create_model'
]

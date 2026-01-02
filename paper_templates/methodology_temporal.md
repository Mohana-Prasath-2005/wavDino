# Methodology Section: Temporal Visual Encoder

## 3. Methodology

### 3.1 Problem Formulation

Given a video clip V with audio A, our goal is to predict the emotion class y ∈ {1, 2, ..., C}. Previous work extracted features from a single video frame, which we extend to temporal multi-frame processing.

### 3.2 Audio Encoder (Unchanged)

We employ Wav2Vec 2.0 [1] as our audio encoder, which learns contextualized representations from raw waveforms through self-supervised learning. Given an audio waveform a ∈ R^T, Wav2Vec 2.0 produces a sequence of features, which we aggregate via mean pooling to obtain:

```
f_audio = Wav2Vec2.0(a) ∈ R^768
```

The pre-trained Wav2Vec 2.0 weights are frozen during training to preserve learned representations and reduce computational cost.

### 3.3 Temporal Visual Encoder (NEW CONTRIBUTION)

**Limitation of Single-Frame Encoding**: In our baseline approach, we extracted features from a single face frame using DINOv2 [2]. While this captures spatial appearance, it fails to model the temporal evolution of emotional expressions.

**Proposed Temporal Modeling**: We extend the visual encoder to process multiple frames with temporal attention:

#### 3.3.1 Frame Sampling

From each video clip V of length T_v frames, we uniformly sample N frames:

```
{I_1, I_2, ..., I_N} where I_i ∈ R^{H×W×3}
```

Frame indices are computed as:

```
idx_i = floor((i / N) × T_v) for i ∈ {0, 1, ..., N-1}
```

We use N=8 for CREMA-D and RAVDESS, and N=16 for AFEW (which contains longer, more dynamic sequences).

#### 3.3.2 Face Detection and Preprocessing

For each sampled frame I_i, we apply:

1. **Face Detection**: MTCNN [3] detects and crops the face region
2. **Alignment**: Faces are aligned and resized to 224×224 pixels
3. **Normalization**: ImageNet normalization (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225])

#### 3.3.3 Frame-Level Feature Extraction

We extract DINOv2 features independently for each frame:

```
f_i = DINOv2(I_i) ∈ R^1024 for i ∈ {1, ..., N}
```

The pre-trained DINOv2 (ViT-L/14) backbone is frozen, ensuring efficient training while leveraging rich visual representations learned from large-scale self-supervised pre-training.

#### 3.3.4 Temporal Attention Pooling

To aggregate frame-level features into a single video-level representation, we employ attention-weighted temporal pooling:

```
α_i = softmax(W_2 · tanh(W_1 · f_i + b_1) + b_2)
```

where W_1 ∈ R^{256×1024}, W_2 ∈ R^{1×256} are learnable projection matrices.

The final visual representation is computed as:

```
f_visual = Σ_{i=1}^N α_i · f_i ∈ R^1024
```

This attention mechanism learns to identify and emphasize frames that are most informative for emotion recognition (e.g., peak emotional expressions).

**Alternative: Temporal Transformer** (Optional Section)

For comparison, we also evaluate a temporal transformer encoder:

```
F = [f_1, f_2, ..., f_N] ∈ R^{N×1024}
F' = TransformerEncoder(F + PE)
f_visual = F'[0, :] (CLS token) or mean(F', dim=0)
```

where PE denotes positional encoding. While this approach offers stronger modeling capacity, it introduces additional parameters. Our ablation studies show that attention pooling provides a better efficiency-performance trade-off.

### 3.4 Multimodal Fusion (Unchanged)

Audio and visual features are fused using a transformer-based fusion module:

```
f_audio_proj = Linear(f_audio) ∈ R^512
f_visual_proj = Linear(f_visual) ∈ R^512

[h_audio, h_visual] = TransformerEncoder([f_audio_proj, f_visual_proj])
h_fused = mean([h_audio, h_visual])
```

### 3.5 Classification Head

The fused representation is passed through an MLP classifier:

```
y_pred = Softmax(MLP(h_fused))
```

where MLP consists of two fully-connected layers with GELU activation and dropout.

### 3.6 Training

**Loss Function**: Cross-entropy loss

```
L = -Σ_{i=1}^C y_i log(ŷ_i)
```

**Optimizer**: AdamW with learning rate 1e-4, weight decay 1e-2

**Learning Rate Schedule**: ReduceLROnPlateau with patience=5, factor=0.5

**Data Augmentation**: 
- Video: Random horizontal flip, color jitter, random rotation (±10°)
- Audio: Additive Gaussian noise (σ=0.005), random amplitude scaling

**Implementation Details**:
- Batch size: 8 (baseline), 6 (8-frame), 4 (16-frame)
- Training epochs: 50 with early stopping (patience=10)
- Hardware: NVIDIA RTX 3090 (24GB)
- Framework: PyTorch 2.0

---

## Key Equations Summary

### Single-Frame Baseline
```
f_visual = DINOv2(I_center)
```

### Temporal (Our Enhancement)
```
f_visual = Σ_{i=1}^N α_i · DINOv2(I_i)
where α_i = softmax(W · tanh(U · DINOv2(I_i)))
```

---

## References to Add

[1] Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", NeurIPS 2020

[2] Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", ICLR 2024

[3] Zhang et al., "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks", IEEE Signal Processing Letters 2016

[4] Vaswani et al., "Attention is All You Need", NeurIPS 2017

[5] Wang et al., "Non-local Neural Networks", CVPR 2018 (for temporal modeling background)

---

## Visual Diagrams to Include

### Figure 1: Architecture Overview
```
Video Clip → [Frame 1, ..., Frame N] → DINOv2 → [f1, ..., fN]
                                                      ↓
                                              Temporal Attention
                                                      ↓
Audio → Wav2Vec 2.0 → f_audio                   f_visual
              ↓                                      ↓
              └─────────→ Fusion Transformer ←───────┘
                                ↓
                          Emotion Classifier
```

### Figure 2: Temporal Attention Mechanism
```
Frame Features:  [f1]  [f2]  [f3]  ...  [fN]
                   ↓     ↓     ↓          ↓
Attention:      [α1]  [α2]  [α3]  ...  [αN]
                   ↓     ↓     ↓          ↓
Weighted Sum: ───→ Σ αi · fi ────→ f_visual
```

---

## Writing Tips

1. **Emphasize the motivation**: Lead with "Why temporal?" before "How temporal?"

2. **Contrast with baseline**: Explicitly state what changed vs. what stayed the same

3. **Justify design choices**: 
   - Why attention pooling over simple averaging? → Learnable weighting
   - Why freeze backbones? → Efficiency and stability
   - Why uniform sampling? → Coverage of entire temporal extent

4. **Ablation preview**: Mention that Section 4.3 validates each design choice

5. **Computational cost**: Add brief analysis showing overhead is minimal:
   - Single frame: 1 × DINOv2 forward pass
   - 8 frames: 8 × DINOv2 forward passes (parallelizable)
   - Attention module: <1M parameters

---

## LaTeX Template (Optional)

```latex
\subsection{Temporal Visual Encoder}

\textbf{Motivation.} Emotional expressions are inherently temporal, evolving from onset through apex to offset. Single-frame visual encoding captures only a momentary snapshot, potentially missing critical dynamics. We address this limitation by extending visual encoding to multi-frame temporal attention.

\textbf{Frame Sampling.} From each video clip of length $T_v$ frames, we uniformly sample $N$ frames:
\begin{equation}
\mathcal{I} = \{I_i\}_{i=1}^N, \quad \text{where } I_i \in \mathbb{R}^{H \times W \times 3}
\end{equation}

\textbf{Temporal Feature Extraction.} Each frame is independently encoded using frozen DINOv2 (ViT-L/14):
\begin{equation}
f_i = \text{DINOv2}(I_i) \in \mathbb{R}^{1024}
\end{equation}

\textbf{Attention-Weighted Aggregation.} To obtain a video-level representation, we compute attention-weighted sum:
\begin{align}
\alpha_i &= \text{softmax}(W_2 \cdot \tanh(W_1 \cdot f_i + b_1) + b_2) \\
f_\text{visual} &= \sum_{i=1}^N \alpha_i \cdot f_i
\end{align}

This allows the model to emphasize frames with salient emotional content.
```

# Neural Context Fusion Network (NCFN)

## Project Summary

The Neural Context Fusion Network (NCFN) is a multimodal machine learning framework designed to act as a contextual decision layer for real-time AI voice generation in live game streaming environments. It processes three simultaneous data streams -- gameplay telemetry, player interaction signals, and speech embeddings -- to produce adaptive voice parameters that drive downstream speech synthesis engines.

NCFN does not replace text-to-speech systems. Instead, it operates as an upstream intelligence layer that determines the optimal voice persona, emotional state, pitch profile, speaking rate, and expressive style before any synthesis occurs. This allows a TTS engine such as ElevenLabs to receive precise, context-aware instructions rather than relying on static voice configurations.

The framework is designed for latency-sensitive streaming contexts where the voice of an AI game commentator or assistant must respond dynamically to rapidly changing game events, audience sentiment, and conversational cues -- producing natural, engaging, and contextually appropriate speech in real time.

---

## Architecture Diagram

```
Microphone
    |
    v
Speech Recognition (Wispr Flow / Whisper)
    |
    v
Speech Embeddings (dim: 256)
    |
    +----------------------------+
    |                            |
Gameplay Telemetry           Live Chat / Community
(health, ammo, kills,        (sentiment, hype, toxicity,
 streak, danger, phase)       velocity, subscriber ratio)
    |                            |
    +------------+---------------+
                 |
                 v
    +------------------------------------------+
    |   Neural Context Fusion Network (NCFN)   |
    |                                          |
    |  [SpeechEncoder]  [GameplayEncoder]      |
    |  [ChatEncoder]                           |
    |       |                                  |
    |       v                                  |
    |  Cross-Modal Attention Fusion            |
    |  (Multi-Head Attention + FFN)            |
    |       |                                  |
    +-------|----------------------------------+
            |
            v
    +------------------------------------------+
    |         Multi-Task Prediction Heads       |
    |                                           |
    |  Persona Head    --> Speaker Identity     |
    |  Emotion Head    --> Emotional State      |
    |  Pitch Head      --> Pitch Shift (Hz)     |
    |  Rate Head       --> Speaking Rate        |
    |  Style Head      --> Expressive Style     |
    +------------------------------------------+
            |
            v
    Voice Parameter JSON
    (persona_id, stability, pitch, rate, style)
            |
            v
    ElevenLabs / Speech Synthesis Engine
            |
            v
    Real-Time AI Voice Output
```

---

## Codebase Structure

| Block | Name | Purpose |
|-------|------|---------|
| 01 | imports_and_config | Imports, hyperparameters (FUSION_DIM=768, NUM_ATTENTION_HEADS=8, BATCH_SIZE=64, etc.), reproducibility seeds, and custom NumPy-based evaluation metrics replacing sklearn |
| 02 | data_generation | Synthetic multimodal dataset generation for 5000 samples across speech embeddings (256-dim), gameplay telemetry (padded to 128-dim), and chat features (padded to 64-dim) with correlated labels for persona, emotion, style, pitch shift, and speaking rate |
| 03 | dataset_and_loaders | NCFNDataset PyTorch class, 80/20 temporal train/val split, and DataLoader construction with shuffling |
| 04 | encoders | Three domain-specific encoder MLPs (SpeechEncoder, GameplayEncoder, ChatEncoder) that project each modality into FUSION_DIM space with attention gating |
| 05 | fusion_layer | CrossModalAttentionFusion layer implementing multi-head attention across the three modality token vectors with residual connections and a position-wise FFN |
| 06 | ncfn_model | Full NCFN model class combining all three encoders, the fusion layer, and five parallel multi-task prediction heads |
| 07 | training | Multi-task training loop with uncertainty-weighted loss (learnable log-variance per head), Adam optimizer, LR scheduler, best-checkpoint saving, and per-epoch metrics logging |
| 08 | evaluation | Full evaluation on the validation set reporting accuracy and macro F1 for classification heads, MAE/RMSE/R2 for regression heads, plus per-class breakdowns and confusion matrices |
| 09 | visualizations | Six-panel training dashboard (loss curves, accuracy curves, regression MAE curves, final accuracy bar chart, pitch scatter plot, speaking rate scatter plot) |
| 10 | inference_pipeline | Real-time inference pipeline that accepts live multimodal inputs and returns an ElevenLabs-compatible voice parameter JSON with persona_id, stability, similarity_boost, style, pitch_shift, speaking_rate, and style_tag |

---

## Key Design Decisions

1. Cross-modal attention fusion was chosen over simple concatenation to allow each modality to dynamically weight and attend to relevant signals in the other modalities. This means gameplay events can suppress or amplify speech features, and chat sentiment can reshape how gameplay context is interpreted -- producing a richer fused representation than any static merging strategy could achieve.

2. Multi-task learning with uncertainty-weighted loss (Kendall uncertainty weighting) is used to automatically balance five heterogeneous prediction targets -- three classification tasks and two regression tasks -- without manually tuning individual loss scale factors. Each task head learns its own log-variance parameter, which the training loop uses to downweight overconfident heads and upweight uncertain ones.

3. Modality-specific encoders are used instead of a shared encoder to preserve domain structure before fusion. Speech, gameplay, and chat data have fundamentally different statistical properties and semantic meanings; projecting each into the shared fusion space through a specialized MLP with attention gating ensures that inter-modality fusion operates on well-structured representations.

4. NumPy-only evaluation metrics are implemented in Block 01 to avoid any dependency on scikit-learn. Functions np_accuracy, np_f1_macro, np_confusion_matrix, np_per_class_stats, np_mae, np_rmse, and np_r2 are available throughout the notebook and produce identical results to their sklearn counterparts.

5. Synthetic data generation with correlated labels is used to produce a realistic training distribution despite the absence of real streaming data. Gameplay intensity is used to drive emotion and pitch variation; health state influences persona assignment; chat sentiment modulates expressive style. These correlations make the training signal meaningful and allow the model to learn coherent cross-modal dependencies.

6. FUSION_DIM=768 was chosen so that 768 divided by 3 equals 256, which is evenly divisible by NUM_ATTENTION_HEADS=8. This ensures that the per-head dimension (32) is a whole number throughout the multi-head attention computation and avoids shape errors in the transformer-style fusion layer.

---

## Input Feature Summary

| Modality | Input Dimension | Key Features |
|----------|-----------------|--------------|
| Speech | 256 | Normalized MFCC-style embeddings, energy, ASR confidence |
| Gameplay | 128 | Health, shield, ammo, kill streak, deaths, distance to objective, game phase (one-hot), event type (one-hot), danger score, teammate count, score delta |
| Chat | 64 | Sentiment score, hype score, toxicity score, chat velocity, subscriber ratio |

---

## Output Parameters

| Head | Type | Labels / Range |
|------|------|----------------|
| Persona | Classification (5 classes) | Commentator, Coach, Villain, Hero, Analyst |
| Emotion | Classification (7 classes) | Neutral, Excited, Tense, Sad, Angry, Fearful, Triumphant |
| Pitch Shift | Regression | -6.0 to +6.0 (semitones) |
| Speaking Rate | Regression | 0.6x to 1.8x |
| Style | Classification (4 classes) | Calm, Intense, Dramatic, Conversational |

---

## ElevenLabs Integration Note

The inference pipeline in Block 10 converts NCFN predictions into ElevenLabs API-compatible voice parameters through a structured mapping layer. The predicted persona class index is mapped to a predefined speaker_id dictionary that associates each persona (Commentator, Coach, Villain, Hero, Analyst) with a specific ElevenLabs voice ID. The predicted emotion and style classes are mapped to stability and similarity_boost values -- with high-energy states such as Excited or Intense producing lower stability and higher similarity_boost for a more expressive output, while calm states produce higher stability. Pitch shift and speaking rate predictions are passed directly to the TTS API call as numeric parameters. The framework is synthesis-engine agnostic: the output is a structured voice parameter JSON that can be routed to any TTS provider with equivalent parameter support.

---

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib

No other dependencies are required. scikit-learn is NOT used; all evaluation metrics are implemented in Block 01 using NumPy.

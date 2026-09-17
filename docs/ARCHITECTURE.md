# Architecture

NCFN is an upstream context layer for AI voice synthesis: it doesn't generate speech itself, it decides *how* speech should sound before a TTS engine (ElevenLabs, or any equivalent) ever runs. It takes three simultaneous streams from a live game stream and fuses them into per-turn voice parameters.

## The three input modalities

| Modality | Dimension | What it carries |
|----------|-----------|-------------------|
| Speech | 256 | MFCC-style embeddings, energy, ASR confidence, from the ASR stage (Wispr Flow / Whisper) upstream of NCFN |
| Gameplay telemetry | 128 (padded) | Health, shield, ammo, kill streak, deaths, distance to objective, game phase, event type, danger score, teammate count, score delta |
| Chat / community | 64 (padded) | Sentiment, hype, toxicity, chat velocity, subscriber ratio |

## Pipeline

```
Microphone
    │
    ▼
Speech Recognition (Wispr Flow / Whisper)
    │
    ▼
Speech Embeddings (256-dim)
    │
    ├───────────────┬───────────────┐
    │                │               │
Speech (256)   Gameplay (128)   Chat (64)
    │                │               │
    ▼                ▼               ▼
SpeechEncoder   GameplayEncoder  ChatEncoder     (each: MLP → FUSION_DIM, with attention gating)
    │                │               │
    └────────┬───────┴───────┬───────┘
             ▼                ▼
       Cross-Modal Attention Fusion
       (multi-head attention + FFN, residual connections)
             │
             ▼
      Multi-Task Prediction Heads
      ├── Persona   (5-class)
      ├── Emotion   (7-class)
      ├── Pitch     (regression, semitones)
      ├── Rate      (regression, speed multiplier)
      └── Style     (4-class)
             │
             ▼
      Voice Parameter JSON
      (persona_id, stability, pitch, rate, style)
             │
             ▼
      ElevenLabs / any equivalent TTS engine
             │
             ▼
      Real-time AI voice output
```

## Why encode each modality separately before fusing

Speech, gameplay, and chat data have fundamentally different statistical properties: one is a continuous embedding from an ASR model, one is a mix of discrete counters and one-hot categorical state, one is a set of sentiment/velocity scores. A shared encoder would force all three into one representation before any modality-specific structure could be preserved. NCFN instead gives each modality its own MLP encoder (with attention gating) to project into the shared `FUSION_DIM` space first, so the fusion layer that follows is operating on three already-well-structured representations rather than one that's had its per-modality structure flattened out prematurely.

## Why cross-modal attention instead of concatenation

Simple concatenation treats all three modalities as a fixed, static combination: whatever weight speech gets relative to gameplay is baked in by the network's early layers and doesn't change per input. Cross-modal attention lets each modality dynamically weight the others per example: a sudden spike in gameplay danger can suppress or amplify how speech features are interpreted, and chat sentiment can reshape how gameplay context gets read. That data-dependent weighting is what concatenation alone can't express.

## Why `FUSION_DIM=768`

768 divides evenly by 3 (one share per modality) into 256, and 256 divides evenly by `NUM_ATTENTION_HEADS=8` into a per-head dimension of 32. Both divisions land on whole numbers, which avoids shape mismatches inside the multi-head attention computation. This is a deliberate dimension choice, not an arbitrary transformer-default value carried over from elsewhere.

## Output: five prediction heads, not one

| Head | Type | Output |
|------|------|--------|
| Persona | Classification (5) | Commentator, Coach, Villain, Hero, Analyst |
| Emotion | Classification (7) | Neutral, Excited, Tense, Sad, Angry, Fearful, Triumphant |
| Pitch Shift | Regression | -6.0 to +6.0 semitones |
| Speaking Rate | Regression | 0.6x to 1.8x |
| Style | Classification (4) | Calm, Intense, Dramatic, Conversational |

Five heads, sharing one fused representation, is what lets a single forward pass produce every parameter a downstream TTS call needs in one shot, rather than requiring five separate models or five separate calls.

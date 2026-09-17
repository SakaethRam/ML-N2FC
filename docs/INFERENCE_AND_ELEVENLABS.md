# Inference and ElevenLabs Integration

## Inference pipeline (Block 10)

Accepts live multimodal input (speech embeddings, gameplay telemetry, chat features) and returns an ElevenLabs-compatible voice parameter JSON:

```json
{
  "persona_id": "...",
  "stability": 0.0,
  "similarity_boost": 0.0,
  "style": 0.0,
  "pitch_shift": 0.0,
  "speaking_rate": 1.0,
  "style_tag": "..."
}
```

## Mapping predictions to TTS parameters

The inference pipeline is doing more than a raw forward pass; it includes a structured mapping layer that converts NCFN's five head outputs into the specific parameters ElevenLabs (or an equivalent engine) expects:

| NCFN output | Mapped to | Mapping logic |
|-------------|-----------|----------------|
| Persona class index | `speaker_id` (via a predefined dictionary) | Each of the 5 personas (Commentator, Coach, Villain, Hero, Analyst) maps to a specific ElevenLabs voice ID. |
| Emotion + Style class | `stability`, `similarity_boost` | High-energy states (Excited, Intense) → lower stability, higher similarity_boost, for a more expressive output. Calm states → higher stability. |
| Pitch Shift (regression) | `pitch_shift` | Passed directly as a numeric parameter. |
| Speaking Rate (regression) | `speaking_rate` | Passed directly as a numeric parameter. |

## Why this is described as "synthesis-engine agnostic"

The output of the inference pipeline is a structured JSON of voice parameters, not an ElevenLabs API call itself embedded in the pipeline. That separation means the persona→voice-ID dictionary and the emotion/style→stability/similarity_boost mapping are the only ElevenLabs-specific pieces; routing the same JSON to a different TTS provider with equivalent parameter support (stability-like and boost-like controls, pitch, rate) would mean replacing that mapping layer, not the model or the fusion architecture.

## What to confirm before wiring this to a live ElevenLabs account

- The `speaker_id` dictionary maps 5 persona classes to specific ElevenLabs voice IDs; those voice IDs are account-specific (cloned or selected voices in your ElevenLabs account), so they need to be created and their IDs substituted in before the pipeline can make a live call, not derived automatically from the model's output.
- Confirm the exact numeric ranges the mapping layer produces for `stability` and `similarity_boost` match ElevenLabs' expected 0.0-1.0 range for those parameters; a regression head's raw output range and the target API's expected range aren't necessarily the same by default.
- Latency: the README's motivating use case (live game commentary) implies this needs to run within a tight latency budget end to end (telemetry in, voice parameters out, TTS audio out). The inference pipeline's own latency isn't benchmarked in the README; profile it directly before assuming it fits a real-time streaming budget.

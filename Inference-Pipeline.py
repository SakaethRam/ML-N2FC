
# ============================================================
# Block 10 — Real-Time Inference Pipeline & ElevenLabs Export
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, Any, Optional

# --- Voice Persona Map ----------------------------------------

PERSONA_TO_VOICE_MAP: Dict[int, Dict[str, Any]] = {
    0: {"name": "Commentator", "voice_id": "commentator_v1",  "base_stability": 0.75},
    1: {"name": "Hero",        "voice_id": "hero_epic_v2",    "base_stability": 0.65},
    2: {"name": "Villain",     "voice_id": "villain_dark_v1", "base_stability": 0.55},
    3: {"name": "Guide",       "voice_id": "guide_calm_v1",   "base_stability": 0.85},
    4: {"name": "Spectator",   "voice_id": "spectator_v1",    "base_stability": 0.70},
}

# ElevenLabs style modulation: maps style_name -> style_intensity
STYLE_INTENSITY_MAP: Dict[str, float] = {
    "narrative":  0.30,
    "intense":    0.85,
    "casual":     0.40,
    "dramatic":   0.95,
}

# Emotions that trigger speaker boost
SPEAKER_BOOST_EMOTIONS = {1, 4}   # excited, triumphant

# Emotion modulation on stability
EMOTION_STABILITY_DELTA: Dict[int, float] = {0: 0.0, 1: -0.05, 2: -0.12, 3: 0.08, 4: -0.03, 5: -0.15}

# similarity_boost per persona
PERSONA_SIMILARITY: Dict[int, float] = {0: 0.80, 1: 0.70, 2: 0.55, 3: 0.85, 4: 0.65}


# --- NCFNInferencePipeline ------------------------------------

class NCFNInferencePipeline:
    """
    End-to-end inference pipeline for NCFN.
    Loads the trained model and converts multimodal inputs to
    ElevenLabs-compatible TTS voice parameters.
    """

    def __init__(self, model_path: str, config):
        self.config = config
        self._device = torch.device(config.DEVICE)

        # Load model from checkpoint
        self._model = NCFNModel(config).to(self._device)
        ckpt = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        print(f"[Pipeline] Model loaded from '{model_path}' "
              f"(best val_loss={ckpt['val_loss']:.4f}, "
              f"trained to epoch {ckpt['epoch']})")

    # Preprocessing

    def _preprocess_speech(self, raw: Dict[str, Any]) -> torch.Tensor:
        D = self.config.SPEECH_EMBED_DIM
        emb = np.array(raw.get("embedding", [0.0] * (D - 2)), dtype=np.float32)
        if len(emb) < D - 2:
            emb = np.pad(emb, (0, D - 2 - len(emb)))
        else:
            emb = emb[:D - 2]
        norm = np.linalg.norm(emb)
        emb  = emb / (norm + 1e-8)
        feat = np.concatenate([emb,
                                [float(raw.get("energy",     0.5)),
                                 float(raw.get("confidence", 0.5))]], axis=0).astype(np.float32)
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self._device)

    def _preprocess_gameplay(self, gs: Dict[str, Any]) -> torch.Tensor:
        _cfg = self.config
        named = np.array([
            float(gs.get("player_health",          50.0)) / 100.0,
            float(gs.get("shield_percent",           0.5)),
            float(gs.get("ammo_ratio",               0.5)),
            float(gs.get("kill_streak",              0.0)) / 20.0,
            float(gs.get("deaths_last_minute",       0.0)) / 5.0,
            float(gs.get("distance_to_objective",  250.0)) / 500.0,
            float(gs.get("time_in_game_minutes",    30.0)) / 60.0,
            float(gs.get("environment_danger",       0.3)),
            float(gs.get("teammate_count",           2.0)) / 4.0,
            float(gs.get("score_delta",              0.0)) / 10.0,
        ], dtype=np.float32)
        phase_oh = np.eye(3, dtype=np.float32)[np.clip(int(gs.get("game_phase", 1)), 0, 2)]
        event_oh = np.eye(5, dtype=np.float32)[np.clip(int(gs.get("event_type", 3)), 0, 4)]
        raw_feat = np.concatenate([named, phase_oh, event_oh])
        pad_len  = _cfg.GAMEPLAY_FEATURE_DIM - len(raw_feat)
        feat     = np.concatenate([raw_feat, np.zeros(pad_len, dtype=np.float32)])
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self._device)

    def _preprocess_chat(self, cs: Dict[str, Any]) -> torch.Tensor:
        _cfg = self.config
        named = np.array([
            float(cs.get("sentiment_score",  0.0)),
            float(cs.get("hype_score",        0.5)),
            float(cs.get("toxicity_score",    0.1)),
            float(cs.get("chat_velocity",    20.0)) / 100.0,
            float(cs.get("subscriber_ratio",  0.3)),
        ], dtype=np.float32)
        pad_len = _cfg.CHAT_EMBED_DIM - len(named)
        feat    = np.concatenate([named, np.zeros(pad_len, dtype=np.float32)])
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self._device)

    # Prediction

    def predict(self, speech_in: Dict, gameplay_in: Dict, chat_in: Dict) -> Dict[str, Any]:
        s_t = self._preprocess_speech(speech_in)
        g_t = self._preprocess_gameplay(gameplay_in)
        c_t = self._preprocess_chat(chat_in)

        with torch.no_grad():
            out = self._model(s_t, g_t, c_t)

        persona_probs = torch.softmax(out["persona_logits"], dim=1).squeeze(0).cpu().numpy()
        emotion_probs = torch.softmax(out["emotion_logits"], dim=1).squeeze(0).cpu().numpy()
        style_probs   = torch.softmax(out["style_logits"],   dim=1).squeeze(0).cpu().numpy()

        persona_idx = int(persona_probs.argmax())
        emotion_idx = int(emotion_probs.argmax())
        style_idx   = int(style_probs.argmax())
        pitch_val   = float(out["pitch_pred"].squeeze().cpu().item())
        rate_val    = float(out["rate_pred"].squeeze().cpu().item())

        return {
            "persona_idx":   persona_idx,
            "persona_name":  self.config.PERSONA_NAMES[persona_idx],
            "persona_probs": persona_probs.tolist(),
            "emotion_idx":   emotion_idx,
            "emotion_name":  self.config.EMOTION_NAMES[emotion_idx],
            "emotion_probs": emotion_probs.tolist(),
            "pitch_shift":   round(float(np.clip(pitch_val, -5.0, 5.0)), 4),
            "speaking_rate": round(float(np.clip(rate_val,   0.6,  1.8)), 4),
            "style_idx":     style_idx,
            "style_name":    self.config.STYLE_NAMES[style_idx],
            "style_probs":   style_probs.tolist(),
        }

    # ElevenLabs Parameter Mapping

    def to_elevenlabs_params(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        persona_idx = pred["persona_idx"]
        emotion_idx = pred["emotion_idx"]
        style_name  = pred["style_name"]
        pitch_shift = pred["pitch_shift"]
        speak_rate  = pred["speaking_rate"]

        voice       = PERSONA_TO_VOICE_MAP[persona_idx]
        stability   = float(np.clip(
            voice["base_stability"] + EMOTION_STABILITY_DELTA.get(emotion_idx, 0.0),
            0.0, 1.0
        ))
        sim_boost       = PERSONA_SIMILARITY[persona_idx]
        style_intensity = STYLE_INTENSITY_MAP.get(style_name, 0.5)
        speaker_boost   = emotion_idx in SPEAKER_BOOST_EMOTIONS
        pitch_norm      = round((pitch_shift + 5.0) / 10.0, 4)

        return {
            "voice_id":              voice["voice_id"],
            "stability":             round(stability, 4),
            "similarity_boost":      round(sim_boost, 4),
            "style":                 round(style_intensity, 4),
            "use_speaker_boost":     speaker_boost,
            "ncfn_persona":          pred["persona_name"],
            "ncfn_emotion":          pred["emotion_name"],
            "ncfn_style":            style_name,
            "ncfn_pitch_shift":      pitch_shift,
            "ncfn_speaking_rate":    speak_rate,
            "ncfn_pitch_normalized": pitch_norm,
        }

    # Full Pipeline

    def run_live(self, speech_in: Dict, gameplay_in: Dict, chat_in: Dict) -> str:
        """Full pipeline: preprocess -> predict -> ElevenLabs params -> JSON string."""
        pred   = self.predict(speech_in, gameplay_in, chat_in)
        params = self.to_elevenlabs_params(pred)
        return json.dumps(params, indent=2)


# --- Instantiate Pipeline -------------------------------------

pipeline = NCFNInferencePipeline(BEST_MODEL_PATH, cfg)

# --- Demo Scenarios -------------------------------------------

# Scenario 1: Boss fight -- high danger, hype chat, low health
scenario_boss_fight = {
    "speech": {
        "embedding":   list(np.random.default_rng(1).standard_normal(cfg.SPEECH_EMBED_DIM - 2)),
        "energy":      0.90,
        "confidence":  0.85,
    },
    "gameplay": {
        "player_health":         15.0,
        "shield_percent":         0.1,
        "ammo_ratio":             0.3,
        "kill_streak":            8.0,
        "deaths_last_minute":     3.0,
        "distance_to_objective": 20.0,
        "time_in_game_minutes":  45.0,
        "environment_danger":     0.9,
        "teammate_count":         1.0,
        "score_delta":           -5.0,
        "game_phase":             2,
        "event_type":             0,
    },
    "chat": {
        "sentiment_score":  0.75,
        "hype_score":        0.95,
        "toxicity_score":    0.05,
        "chat_velocity":    85.0,
        "subscriber_ratio":  0.4,
    },
}

# Scenario 2: Calm exploration -- full health, quiet chat
scenario_calm_explore = {
    "speech": {
        "embedding":   list(np.random.default_rng(2).standard_normal(cfg.SPEECH_EMBED_DIM - 2)),
        "energy":      0.30,
        "confidence":  0.92,
    },
    "gameplay": {
        "player_health":         95.0,
        "shield_percent":         0.9,
        "ammo_ratio":             0.85,
        "kill_streak":            0.0,
        "deaths_last_minute":     0.0,
        "distance_to_objective": 400.0,
        "time_in_game_minutes":   5.0,
        "environment_danger":     0.1,
        "teammate_count":         3.0,
        "score_delta":            2.0,
        "game_phase":             0,
        "event_type":             3,
    },
    "chat": {
        "sentiment_score":  0.20,
        "hype_score":        0.15,
        "toxicity_score":    0.02,
        "chat_velocity":    10.0,
        "subscriber_ratio":  0.6,
    },
}

print("=" * 60)
print("  NCFN Inference Pipeline Demo")
print("=" * 60)

for _name, _scenario in [("Boss Fight", scenario_boss_fight), ("Calm Exploration", scenario_calm_explore)]:
    print(f"\n[Scenario] {_name}")
    print("-" * 40)
    _result = pipeline.run_live(
        _scenario["speech"],
        _scenario["gameplay"],
        _scenario["chat"],
    )
    print(_result)

print("\n[Block 10] Real-time inference pipeline -- DONE")

# Expose a sample output as a named variable
sample_voice_params = json.loads(pipeline.run_live(
    scenario_boss_fight["speech"],
    scenario_boss_fight["gameplay"],
    scenario_boss_fight["chat"],
))

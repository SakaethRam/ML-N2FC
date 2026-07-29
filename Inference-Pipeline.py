
# ============================================================
# Block 10 — Real-Time Inference Pipeline & ElevenLabs Export
# ============================================================

# ─── Voice Persona Map ────────────────────────────────────────

PERSONA_TO_VOICE_MAP: Dict[int, Dict[str, Any]] = {
    0: {"name": "Commentator", "voice_id": "commentator_v1",  "base_stability": 0.75},
    1: {"name": "Hero",        "voice_id": "hero_epic_v2",    "base_stability": 0.65},
    2: {"name": "Villain",     "voice_id": "villain_dark_v1", "base_stability": 0.55},
    3: {"name": "Guide",       "voice_id": "guide_calm_v1",   "base_stability": 0.85},
    4: {"name": "Spectator",   "voice_id": "spectator_v1",    "base_stability": 0.70},
}

# ElevenLabs style modulation: maps (style_idx, emotion_idx) → style_intensity
STYLE_INTENSITY_MAP: Dict[str, float] = {
    "narrative":  0.30,
    "intense":    0.85,
    "casual":     0.40,
    "dramatic":   0.95,
}

# Emotions that trigger speaker boost
SPEAKER_BOOST_EMOTIONS: set = {1, 4}   # excited, triumphant


# ─── NCFNInferencePipeline ────────────────────────────────────

class NCFNInferencePipeline:
    """
    End-to-end inference pipeline for NCFN.
    Loads the trained model and converts multimodal inputs to
    ElevenLabs-compatible TTS voice parameters.
    """

    def __init__(self, model_path: str, config: NCFNConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Load model from checkpoint
        self._model = NCFNModel(config).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        print(f"[Pipeline] Model loaded from '{model_path}' "
              f"(best val_loss={ckpt['val_loss']:.4f}, "
              f"trained to epoch {ckpt['epoch']})")

    # ── Preprocessing ─────────────────────────────────────────

    def _preprocess_speech(self, raw_audio_features: Dict[str, Any]) -> torch.Tensor:
        """
        Accepts dict with keys:
            embedding : list/array of (SPEECH_EMBED_DIM - 2) floats (Whisper output)
            energy    : float 0-1
            confidence: float 0-1
        Returns normalized tensor [1, SPEECH_EMBED_DIM].
        """
        D = self.config.SPEECH_EMBED_DIM
        embedding  = np.array(raw_audio_features.get("embedding", [0.0] * (D - 2)), dtype=np.float32)

        # Pad or truncate to exactly D-2 dims
        if len(embedding) < D - 2:
            embedding = np.pad(embedding, (0, D - 2 - len(embedding)))
        else:
            embedding = embedding[:D - 2]

        # L2-normalize
        norm = np.linalg.norm(embedding)
        embedding = embedding / (norm + 1e-8)

        energy     = float(raw_audio_features.get("energy", 0.5))
        confidence = float(raw_audio_features.get("confidence", 0.5))

        feat = np.concatenate([embedding, [energy, confidence]], axis=0).astype(np.float32)
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)   # [1, D]

    def _preprocess_gameplay(self, gameplay_state: Dict[str, Any]) -> torch.Tensor:
        """
        Accepts structured dict with named gameplay fields.
        Returns tensor [1, GAMEPLAY_FEATURE_DIM].
        """
        cfg = self.config

        player_health      = float(gameplay_state.get("player_health",      50.0)) / 100.0
        shield_percent     = float(gameplay_state.get("shield_percent",       0.5))
        ammo_ratio         = float(gameplay_state.get("ammo_ratio",           0.5))
        kill_streak        = float(gameplay_state.get("kill_streak",          0.0)) / 20.0
        deaths_last_minute = float(gameplay_state.get("deaths_last_minute",   0.0)) / 5.0
        dist_to_objective  = float(gameplay_state.get("distance_to_objective",250.0)) / 500.0
        time_in_game       = float(gameplay_state.get("time_in_game_minutes", 30.0)) / 60.0
        environment_danger = float(gameplay_state.get("environment_danger",    0.3))
        teammate_count     = float(gameplay_state.get("teammate_count",        2.0)) / 4.0
        score_delta        = float(gameplay_state.get("score_delta",           0.0)) / 10.0

        # game_phase one-hot
        game_phase_idx = int(gameplay_state.get("game_phase", 1))   # 0/1/2
        game_phase_onehot = np.eye(3, dtype=np.float32)[np.clip(game_phase_idx, 0, 2)]

        # event_type one-hot
        event_type_idx = int(gameplay_state.get("event_type", 3))   # 0-4
        event_type_onehot = np.eye(5, dtype=np.float32)[np.clip(event_type_idx, 0, 4)]

        named = np.array([
            player_health, shield_percent, ammo_ratio, kill_streak,
            deaths_last_minute, dist_to_objective, time_in_game,
            environment_danger, teammate_count, score_delta,
        ], dtype=np.float32)  # 10 scalars

        feat_list = np.concatenate([named, game_phase_onehot, event_type_onehot])
        # Pad to GAMEPLAY_FEATURE_DIM with zeros
        NAMED_TOTAL = len(feat_list)  # 10 + 3 + 5 = 18
        pad_len = cfg.GAMEPLAY_FEATURE_DIM - NAMED_TOTAL
        feat = np.concatenate([feat_list, np.zeros(pad_len, dtype=np.float32)])

        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)   # [1, D]

    def _preprocess_chat(self, chat_state: Dict[str, Any]) -> torch.Tensor:
        """
        Accepts structured chat dict.
        Returns tensor [1, CHAT_EMBED_DIM].
        """
        cfg = self.config

        sentiment_score  = float(chat_state.get("sentiment_score",  0.0))
        hype_score       = float(chat_state.get("hype_score",        0.5))
        toxicity_score   = float(chat_state.get("toxicity_score",    0.1))
        chat_velocity    = float(chat_state.get("chat_velocity",    20.0)) / 100.0
        subscriber_ratio = float(chat_state.get("subscriber_ratio",  0.3))

        named = np.array([
            sentiment_score, hype_score, toxicity_score,
            chat_velocity, subscriber_ratio,
        ], dtype=np.float32)

        pad_len = cfg.CHAT_EMBED_DIM - len(named)
        feat = np.concatenate([named, np.zeros(pad_len, dtype=np.float32)])

        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)   # [1, D]

    # ── Prediction ────────────────────────────────────────────

    def predict(
        self,
        speech_input:   Dict[str, Any],
        gameplay_input: Dict[str, Any],
        chat_input:     Dict[str, Any],
    ) -> Dict[str, Any]:
        """Runs full NCFN forward pass. Returns raw predictions dict."""
        speech_t   = self._preprocess_speech(speech_input)
        gameplay_t = self._preprocess_gameplay(gameplay_input)
        chat_t     = self._preprocess_chat(chat_input)

        with torch.no_grad():
            raw_out = self._model(speech_t, gameplay_t, chat_t)

        persona_probs  = torch.softmax(raw_out["persona_logits"],  dim=1).squeeze(0).cpu().numpy()
        emotion_probs  = torch.softmax(raw_out["emotion_logits"],  dim=1).squeeze(0).cpu().numpy()
        style_probs    = torch.softmax(raw_out["style_logits"],    dim=1).squeeze(0).cpu().numpy()

        persona_idx = int(persona_probs.argmax())
        emotion_idx = int(emotion_probs.argmax())
        style_idx   = int(style_probs.argmax())
        pitch_val   = float(raw_out["pitch_pred"].squeeze().cpu().item())
        rate_val    = float(raw_out["rate_pred"].squeeze().cpu().item())

        return {
            "persona_idx":    persona_idx,
            "persona_name":   self.config.PERSONA_NAMES[persona_idx],
            "persona_probs":  persona_probs.tolist(),
            "emotion_idx":    emotion_idx,
            "emotion_name":   self.config.EMOTION_NAMES[emotion_idx],
            "emotion_probs":  emotion_probs.tolist(),
            "pitch_shift":    round(float(np.clip(pitch_val, -5.0, 5.0)), 4),
            "speaking_rate":  round(float(np.clip(rate_val,  0.6,  1.8)), 4),
            "style_idx":      style_idx,
            "style_name":     self.config.STYLE_NAMES[style_idx],
            "style_probs":    style_probs.tolist(),
        }

    # ── ElevenLabs Parameter Mapping ─────────────────────────

    def to_elevenlabs_params(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps NCFN outputs to ElevenLabs-compatible voice parameters.

        stability        : base + emotion modulation (high danger → lower stability)
        similarity_boost : persona consistency (Commentator → high, Villain → low)
        style            : 0-1 style intensity from style prediction
        use_speaker_boost: True if emotion is excited or triumphant
        voice_id         : from PERSONA_TO_VOICE_MAP
        """
        persona_idx  = predictions["persona_idx"]
        emotion_idx  = predictions["emotion_idx"]
        style_name   = predictions["style_name"]
        pitch_shift  = predictions["pitch_shift"]
        speak_rate   = predictions["speaking_rate"]

        persona_voice = PERSONA_TO_VOICE_MAP[persona_idx]
        base_stability = persona_voice["base_stability"]

        # Emotion modulation: tense/fearful → reduce stability; calm → increase
        EMOTION_STABILITY_DELTA = {0: 0.0, 1: -0.05, 2: -0.12, 3: 0.08, 4: -0.03, 5: -0.15}
        stability = float(np.clip(
            base_stability + EMOTION_STABILITY_DELTA.get(emotion_idx, 0.0),
            0.0, 1.0
        ))

        # similarity_boost: higher for Commentator/Guide (consistent voice), lower for Villain
        PERSONA_SIMILARITY = {0: 0.80, 1: 0.70, 2: 0.55, 3: 0.85, 4: 0.65}
        similarity_boost = PERSONA_SIMILARITY[persona_idx]

        # Style intensity
        style_intensity = STYLE_INTENSITY_MAP.get(style_name, 0.5)

        # Speaker boost for high-energy emotions
        use_speaker_boost = emotion_idx in SPEAKER_BOOST_EMOTIONS

        # Normalize pitch & rate for ElevenLabs (informational — passed as custom metadata)
        pitch_semitones_normalized = round((pitch_shift + 5.0) / 10.0, 4)   # 0-1

        return {
            # ElevenLabs API parameters
            "voice_id":          persona_voice["voice_id"],
            "stability":         round(stability, 4),
            "similarity_boost":  round(similarity_boost, 4),
            "style":             round(style_intensity, 4),
            "use_speaker_boost": use_speaker_boost,
            # NCFN predictions (for logging/debugging)
            "ncfn_persona":      predictions["persona_name"],
            "ncfn_emotion":      predictions["emotion_name"],
            "ncfn_style":        style_name,
            "ncfn_pitch_shift":  pitch_shift,
            "ncfn_speaking_rate": speak_rate,
            "ncfn_pitch_normalized": pitch_semitones_normalized,
        }

    # ── Full Pipeline ─────────────────────────────────────────

    def run_live(
        self,
        speech_input:   Dict[str, Any],
        gameplay_input: Dict[str, Any],
        chat_input:     Dict[str, Any],
    ) -> str:
        """Full pipeline: preprocess → predict → ElevenLabs params → JSON string."""
        predictions = self.predict(speech_input, gameplay_input, chat_input)
        elevenlabs_params = self.to_elevenlabs_params(predictions)
        return json.dumps(elevenlabs_params, indent=2)


# ─── Instantiate Pipeline ─────────────────────────────────────

pipeline = NCFNInferencePipeline(BEST_MODEL_PATH, cfg)

# ─── Demo Scenarios ───────────────────────────────────────────

# Scenario 1: Boss fight — high danger, hype chat, low health, excited streamer
scenario_boss_fight = {
    "speech": {
        "embedding":   list(np.random.default_rng(1).standard_normal(cfg.SPEECH_EMBED_DIM - 2)),
        "energy":      0.90,
        "confidence":  0.85,
    },
    "gameplay": {
        "player_health":       15.0,
        "shield_percent":      0.1,
        "ammo_ratio":          0.3,
        "kill_streak":         8.0,
        "deaths_last_minute":  3.0,
        "distance_to_objective": 20.0,
        "time_in_game_minutes":  45.0,
        "game_phase":          2,          # late
        "event_type":          4,          # boss_fight
        "environment_danger":  0.95,
        "teammate_count":      1.0,
        "score_delta":         5.0,
    },
    "chat": {
        "sentiment_score":  0.85,
        "hype_score":       0.95,
        "toxicity_score":   0.05,
        "chat_velocity":    85.0,
        "subscriber_ratio": 0.60,
    },
}

# Scenario 2: Idle exploration — calm, low chat, medium health
scenario_idle = {
    "speech": {
        "embedding":   list(np.random.default_rng(2).standard_normal(cfg.SPEECH_EMBED_DIM - 2)),
        "energy":      0.30,
        "confidence":  0.70,
    },
    "gameplay": {
        "player_health":       70.0,
        "shield_percent":      0.8,
        "ammo_ratio":          0.9,
        "kill_streak":         0.0,
        "deaths_last_minute":  0.0,
        "distance_to_objective": 300.0,
        "time_in_game_minutes":   15.0,
        "game_phase":          0,          # early
        "event_type":          3,          # idle
        "environment_danger":  0.10,
        "teammate_count":      3.0,
        "score_delta":         0.0,
    },
    "chat": {
        "sentiment_score":  0.20,
        "hype_score":       0.15,
        "toxicity_score":   0.02,
        "chat_velocity":    8.0,
        "subscriber_ratio": 0.40,
    },
}

# ─── Run Both Scenarios ───────────────────────────────────────

print("=" * 60)
print("  NCFN Inference Pipeline — Demo")
print("=" * 60)

print("\n━━━ Scenario 1: Boss Fight (High Danger, High Hype, Low HP) ━━━")
result_boss = pipeline.run_live(
    scenario_boss_fight["speech"],
    scenario_boss_fight["gameplay"],
    scenario_boss_fight["chat"],
)
print(result_boss)

print("\n━━━ Scenario 2: Idle Exploration (Calm, Low Chat, Medium HP) ━━━")
result_idle = pipeline.run_live(
    scenario_idle["speech"],
    scenario_idle["gameplay"],
    scenario_idle["chat"],
)
print(result_idle)

# ─── Parameter Change Summary ────────────────────────────────

boss_params = json.loads(result_boss)
idle_params = json.loads(result_idle)

print("\n━━━ Parameter Delta: Boss Fight vs Idle Exploration ━━━")
print(f"{'Parameter':<26}  {'Boss Fight':<22}  {'Idle Exploration'}")
print("─" * 72)

compared_keys = [
    "voice_id", "stability", "similarity_boost", "style",
    "use_speaker_boost", "ncfn_persona", "ncfn_emotion", "ncfn_style",
    "ncfn_pitch_shift", "ncfn_speaking_rate",
]

for key in compared_keys:
    boss_val = boss_params.get(key, "—")
    idle_val = idle_params.get(key, "—")
    changed  = "◄ CHANGED" if boss_val != idle_val else ""
    print(f"  {key:<26} {str(boss_val):<22}  {str(idle_val):<20}  {changed}")

print("\n[Block 10] Real-time inference pipeline & ElevenLabs export — DONE")
print("\n" + "=" * 60)
print("  Neural Context Fusion Network (NCFN) — All 10 Blocks Ready")
print("=" * 60)

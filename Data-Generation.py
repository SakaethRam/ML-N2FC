
# ============================================================
# Block 2 — Synthetic Multimodal Dataset Generation
# ============================================================

import numpy as np
import pandas as pd

rng = np.random.default_rng(cfg.RANDOM_SEED)
N = cfg.NUM_SAMPLES

# ── Speech Embeddings (SPEECH_EMBED_DIM = 256) ──────────────

# Raw embeddings from N(0,1), then L2-normalized
speech_raw = rng.standard_normal((N, cfg.SPEECH_EMBED_DIM - 2)).astype(np.float32)
norms = np.linalg.norm(speech_raw, axis=1, keepdims=True)
speech_embeddings = speech_raw / (norms + 1e-8)

# Scalar features appended to make final dim = 256
energy      = rng.uniform(0, 1, (N, 1)).astype(np.float32)       # volume/energy
confidence  = rng.uniform(0, 1, (N, 1)).astype(np.float32)       # speaking confidence

speech_features = np.concatenate([speech_embeddings, energy, confidence], axis=1)
assert speech_features.shape == (N, cfg.SPEECH_EMBED_DIM), \
    f"Speech shape mismatch: {speech_features.shape}"

# ── Gameplay Telemetry (GAMEPLAY_FEATURE_DIM = 48) ──────────

player_health          = rng.uniform(0, 100, (N, 1)).astype(np.float32)
shield_percent         = rng.uniform(0, 1,   (N, 1)).astype(np.float32)
ammo_ratio             = rng.uniform(0, 1,   (N, 1)).astype(np.float32)
kill_streak            = rng.uniform(0, 20,  (N, 1)).astype(np.float32)
deaths_last_minute     = rng.uniform(0, 5,   (N, 1)).astype(np.float32)
dist_to_objective      = rng.uniform(0, 500, (N, 1)).astype(np.float32)
time_in_game           = rng.uniform(0, 60,  (N, 1)).astype(np.float32)

# game_phase: 0=early, 1=mid, 2=late → one-hot (3 dims)
game_phase_raw = rng.integers(0, 3, size=N)
game_phase_onehot = np.eye(3, dtype=np.float32)[game_phase_raw]

# event_type: 0=kill, 1=death, 2=objective, 3=idle, 4=boss_fight → one-hot (5 dims)
event_type_raw = rng.integers(0, 5, size=N)
event_type_onehot = np.eye(5, dtype=np.float32)[event_type_raw]

environment_danger = rng.uniform(0, 1, (N, 1)).astype(np.float32)
teammate_count     = rng.uniform(0, 4, (N, 1)).astype(np.float32)
score_delta        = rng.uniform(-10, 10, (N, 1)).astype(np.float32)

# Named scalar dims so far: 7 scalars + 3 game_phase + 5 event_type + 3 scalars = 18 dims
# Remainder: 48 - 18 = 30 random dims
GAMEPLAY_NAMED_DIMS = 18
GAMEPLAY_PAD_DIMS = cfg.GAMEPLAY_FEATURE_DIM - GAMEPLAY_NAMED_DIMS
gameplay_pad = rng.standard_normal((N, GAMEPLAY_PAD_DIMS)).astype(np.float32) * 0.1

gameplay_features = np.concatenate([
    player_health / 100.0,    # normalize to [0,1]
    shield_percent,
    ammo_ratio,
    kill_streak / 20.0,
    deaths_last_minute / 5.0,
    dist_to_objective / 500.0,
    time_in_game / 60.0,
    game_phase_onehot,
    event_type_onehot,
    environment_danger,
    teammate_count / 4.0,
    score_delta / 10.0,
    gameplay_pad,
], axis=1).astype(np.float32)
assert gameplay_features.shape == (N, cfg.GAMEPLAY_FEATURE_DIM), \
    f"Gameplay shape mismatch: {gameplay_features.shape}"

# ── Chat & Sentiment (CHAT_EMBED_DIM = 128) ─────────────────

sentiment_score  = rng.uniform(-1, 1,   (N, 1)).astype(np.float32)
hype_score       = rng.uniform(0, 1,    (N, 1)).astype(np.float32)
toxicity_score   = rng.uniform(0, 1,    (N, 1)).astype(np.float32)
chat_velocity    = rng.uniform(0, 100,  (N, 1)).astype(np.float32)
subscriber_ratio = rng.uniform(0, 1,    (N, 1)).astype(np.float32)

# Named: 5 dims; remainder: 128 - 5 = 123 random dims
CHAT_NAMED_DIMS = 5
CHAT_PAD_DIMS = cfg.CHAT_EMBED_DIM - CHAT_NAMED_DIMS
chat_pad = rng.standard_normal((N, CHAT_PAD_DIMS)).astype(np.float32) * 0.1

chat_features = np.concatenate([
    sentiment_score,
    hype_score,
    toxicity_score,
    chat_velocity / 100.0,
    subscriber_ratio,
    chat_pad,
], axis=1).astype(np.float32)
assert chat_features.shape == (N, cfg.CHAT_EMBED_DIM), \
    f"Chat shape mismatch: {chat_features.shape}"

# ── Derived Signals for Realistic Labels ─────────────────────

# Flatten raw scalars for label logic
hype_flat       = hype_score.squeeze()
energy_flat     = energy.squeeze()
kill_flat       = kill_streak.squeeze() / 20.0
health_flat     = player_health.squeeze() / 100.0
danger_flat     = environment_danger.squeeze()
event_flat      = event_type_raw             # 0-4 integer
game_phase_flat = game_phase_raw             # 0-2 integer
sentiment_flat  = sentiment_score.squeeze()
chat_vel_flat   = chat_velocity.squeeze() / 100.0

# ── Label Generation ─────────────────────────────────────────

# --- EMOTION (0-5: neutral, excited, tense, calm, triumphant, fearful) ---
emotion_scores = np.zeros((N, 6), dtype=np.float32)
emotion_scores[:, 0] += 0.5 - hype_flat * 0.3 - energy_flat * 0.2             # neutral
emotion_scores[:, 1] += kill_flat * 0.6 + hype_flat * 0.5 + energy_flat * 0.4 # excited
emotion_scores[:, 2] += danger_flat * 0.5 + (1 - health_flat) * 0.4           # tense
emotion_scores[:, 3] += (1 - danger_flat) * 0.4 + (1 - hype_flat) * 0.3       # calm
emotion_scores[:, 4] += kill_flat * 0.7 + sentiment_flat * 0.3 + hype_flat * 0.4  # triumphant
emotion_scores[:, 5] += (1 - health_flat) * 0.5 + danger_flat * 0.5            # fearful
# Boss fight boost
emotion_scores[event_flat == 4, 5] += 0.8
emotion_scores[event_flat == 4, 2] += 0.5
# Add noise
emotion_scores += rng.uniform(0, 0.3, emotion_scores.shape)
emotion_label = emotion_scores.argmax(axis=1).astype(np.int64)

# --- STYLE (0-3: narrative, intense, casual, dramatic) ---
style_scores = np.zeros((N, 4), dtype=np.float32)
style_scores[:, 0] += (1 - hype_flat) * 0.5 + (1 - energy_flat) * 0.3          # narrative
style_scores[:, 1] += kill_flat * 0.5 + danger_flat * 0.4 + hype_flat * 0.4    # intense
style_scores[:, 2] += (1 - danger_flat) * 0.3 + (1 - kill_flat) * 0.3          # casual
style_scores[:, 3] += (event_flat == 4).astype(float) * 0.9 + energy_flat * 0.3  # dramatic
style_scores += rng.uniform(0, 0.3, style_scores.shape)
style_label = style_scores.argmax(axis=1).astype(np.int64)

# --- PERSONA (0-4: Commentator, Hero, Villain, Guide, Spectator) ---
persona_scores = np.zeros((N, 5), dtype=np.float32)
persona_scores[:, 0] += chat_vel_flat * 0.5 + hype_flat * 0.4                  # Commentator
persona_scores[:, 1] += kill_flat * 0.5 + health_flat * 0.3 + energy_flat * 0.3  # Hero
persona_scores[:, 2] += (1 - health_flat) * 0.3 + danger_flat * 0.4            # Villain
persona_scores[:, 3] += (1 - danger_flat) * 0.4 + (1 - hype_flat) * 0.3        # Guide
persona_scores[:, 4] += (1 - energy_flat) * 0.4 + (1 - chat_vel_flat) * 0.3    # Spectator
persona_scores += rng.uniform(0, 0.3, persona_scores.shape)
persona_label = persona_scores.argmax(axis=1).astype(np.int64)

# --- PITCH SHIFT: -5.0 to +5.0 semitones (correlated with emotion/energy) ---
emotion_pitch_delta = np.array([ 0.0,  3.0, -2.0, -1.5,  4.0, -3.5], dtype=np.float32)
pitch_shift = (
    emotion_pitch_delta[emotion_label]
    + energy_flat * 2.0
    + rng.uniform(-1.5, 1.5, N).astype(np.float32)
)
pitch_shift = np.clip(pitch_shift, -5.0, 5.0).astype(np.float32)

# --- SPEAKING RATE: 0.6 to 1.8x (correlated with gameplay intensity) ---
intensity = (kill_flat * 0.4 + danger_flat * 0.3 + hype_flat * 0.3).astype(np.float32)
speaking_rate = (
    0.6 + intensity * 1.2
    + rng.uniform(-0.15, 0.15, N).astype(np.float32)
)
speaking_rate = np.clip(speaking_rate, 0.6, 1.8).astype(np.float32)

# ── Build DataFrame ──────────────────────────────────────────

speech_cols   = [f"speech_{i}" for i in range(cfg.SPEECH_EMBED_DIM)]
gameplay_cols = [f"gameplay_{i}" for i in range(cfg.GAMEPLAY_FEATURE_DIM)]
chat_cols     = [f"chat_{i}" for i in range(cfg.CHAT_EMBED_DIM)]

df_speech   = pd.DataFrame(speech_features,   columns=speech_cols)
df_gameplay = pd.DataFrame(gameplay_features, columns=gameplay_cols)
df_chat     = pd.DataFrame(chat_features,     columns=chat_cols)

df_labels = pd.DataFrame({
    "persona_label":  persona_label,
    "emotion_label":  emotion_label,
    "pitch_shift":    pitch_shift,
    "speaking_rate":  speaking_rate,
    "style_label":    style_label,
})

df = pd.concat([df_speech, df_gameplay, df_chat, df_labels], axis=1)

# ── Print Summary ─────────────────────────────────────────────

print("=" * 60)
print("  Synthetic Dataset Generation Summary")
print("=" * 60)
print(f"\n[Shape] df.shape = {df.shape}")
print(f"\n[Feature Dims]")
print(f"  Speech features   : {speech_features.shape}")
print(f"  Gameplay features : {gameplay_features.shape}")
print(f"  Chat features     : {chat_features.shape}")
print(f"\n[Label Distributions]")
print(f"  persona_label  — counts: {np.bincount(persona_label)}")
print(f"  emotion_label  — counts: {np.bincount(emotion_label)}")
print(f"  style_label    — counts: {np.bincount(style_label)}")
print(f"  pitch_shift    — min: {pitch_shift.min():.2f}, max: {pitch_shift.max():.2f}, "
      f"mean: {pitch_shift.mean():.2f}")
print(f"  speaking_rate  — min: {speaking_rate.min():.2f}, max: {speaking_rate.max():.2f}, "
      f"mean: {speaking_rate.mean():.2f}")
print(f"\n[Preview] First 5 rows (label columns):")
print(df[["persona_label", "emotion_label", "pitch_shift", "speaking_rate", "style_label"]].head())
print("\n[Block 2] Synthetic multimodal dataset generation — DONE")

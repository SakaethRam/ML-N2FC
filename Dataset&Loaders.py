
# ============================================================
# Block 3 — NCFNDataset & DataLoaders
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NCFNDataset(Dataset):
    """
    Multimodal dataset for the Neural Context Fusion Network.
    Returns a dict of tensors per sample.
    """

    def __init__(
        self,
        speech_arr:   np.ndarray,
        gameplay_arr: np.ndarray,
        chat_arr:     np.ndarray,
        persona_arr:  np.ndarray,
        emotion_arr:  np.ndarray,
        pitch_arr:    np.ndarray,
        rate_arr:     np.ndarray,
        style_arr:    np.ndarray,
    ):
        self.speech   = torch.tensor(speech_arr,   dtype=torch.float32)
        self.gameplay = torch.tensor(gameplay_arr, dtype=torch.float32)
        self.chat     = torch.tensor(chat_arr,     dtype=torch.float32)

        self.persona_label = torch.tensor(persona_arr, dtype=torch.long)
        self.emotion_label = torch.tensor(emotion_arr, dtype=torch.long)
        self.pitch_shift   = torch.tensor(pitch_arr,   dtype=torch.float32)
        self.speaking_rate = torch.tensor(rate_arr,    dtype=torch.float32)
        self.style_label   = torch.tensor(style_arr,   dtype=torch.long)

    def __len__(self) -> int:
        return len(self.speech)

    def __getitem__(self, idx: int):
        return {
            "speech":       self.speech[idx],
            "gameplay":     self.gameplay[idx],
            "chat":         self.chat[idx],
            "persona_label": self.persona_label[idx],
            "emotion_label": self.emotion_label[idx],
            "pitch_shift":   self.pitch_shift[idx],
            "speaking_rate": self.speaking_rate[idx],
            "style_label":   self.style_label[idx],
        }


# ── Extract arrays from DataFrame ───────────────────────────

speech_cols_b3   = [c for c in df.columns if c.startswith("speech_")]
gameplay_cols_b3 = [c for c in df.columns if c.startswith("gameplay_")]
chat_cols_b3     = [c for c in df.columns if c.startswith("chat_")]

speech_arr   = df[speech_cols_b3].values.astype(np.float32)
gameplay_arr = df[gameplay_cols_b3].values.astype(np.float32)
chat_arr     = df[chat_cols_b3].values.astype(np.float32)
persona_arr  = df["persona_label"].values.astype(np.int64)
emotion_arr  = df["emotion_label"].values.astype(np.int64)
pitch_arr    = df["pitch_shift"].values.astype(np.float32)
rate_arr     = df["speaking_rate"].values.astype(np.float32)
style_arr    = df["style_label"].values.astype(np.int64)

# ── Temporal Split (first 80% = train, last 20% = val) ──────

SPLIT_IDX = int(cfg.NUM_SAMPLES * cfg.TRAIN_SPLIT)

train_dataset = NCFNDataset(
    speech_arr[:SPLIT_IDX],
    gameplay_arr[:SPLIT_IDX],
    chat_arr[:SPLIT_IDX],
    persona_arr[:SPLIT_IDX],
    emotion_arr[:SPLIT_IDX],
    pitch_arr[:SPLIT_IDX],
    rate_arr[:SPLIT_IDX],
    style_arr[:SPLIT_IDX],
)

val_dataset = NCFNDataset(
    speech_arr[SPLIT_IDX:],
    gameplay_arr[SPLIT_IDX:],
    chat_arr[SPLIT_IDX:],
    persona_arr[SPLIT_IDX:],
    emotion_arr[SPLIT_IDX:],
    pitch_arr[SPLIT_IDX:],
    rate_arr[SPLIT_IDX:],
    style_arr[SPLIT_IDX:],
)

# ── DataLoaders ──────────────────────────────────────────────

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(cfg.DEVICE == "cuda"),
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(cfg.DEVICE == "cuda"),
    drop_last=False,
)

# ── Verify a Sample Batch ─────────────────────────────────────

sample_batch = next(iter(train_loader))

print("=" * 60)
print("  Dataset & DataLoader Summary")
print("=" * 60)
print(f"\n[Split]  Total samples  : {cfg.NUM_SAMPLES}")
print(f"         Train samples  : {len(train_dataset)}")
print(f"         Val samples    : {len(val_dataset)}")
print(f"         Train batches  : {len(train_loader)}")
print(f"         Val batches    : {len(val_loader)}")
print(f"\n[Sample Batch] Keys & Shapes:")
for key, tensor in sample_batch.items():
    print(f"  {key:<18} → shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}")
print("\n[Block 3] NCFNDataset and DataLoaders — DONE")


# ============================================================
# Block 1 — Imports, Config & Reproducibility
# ============================================================

import os
import json
import time
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# Try to import seaborn; if unavailable, create a stub
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    class _SnsStub:
        def heatmap(self, *args, **kwargs):
            ax = kwargs.get('ax', plt.gca())
            data = args[0] if args else kwargs.get('data')
            if data is not None:
                ax.imshow(data, aspect='auto', cmap='Blues')
            return ax
        def set_theme(self, *args, **kwargs): pass
        def set_palette(self, *args, **kwargs): pass
    sns = _SnsStub()
    print("[Warning] seaborn not available; using matplotlib fallback for heatmaps")

# ─── Global Config Dataclass ────────────────────────────────

@dataclass
class NCFNConfig:
    # Input dimensions
    SPEECH_EMBED_DIM: int = 256
    GAMEPLAY_FEATURE_DIM: int = 48
    CHAT_EMBED_DIM: int = 128

    # Model architecture
    FUSION_DIM: int = 768  # Changed from 512 to 768: 768//3=256, 256%8=0 ✓
    NUM_ATTENTION_HEADS: int = 8

    # Output classes
    NUM_PERSONAS: int = 5
    NUM_EMOTIONS: int = 6
    NUM_STYLES: int = 4

    # Training hyperparameters
    DROPOUT_RATE: float = 0.2
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 40
    TRAIN_SPLIT: float = 0.8

    # Data
    NUM_SAMPLES: int = 5000
    RANDOM_SEED: int = 42

    # Device (computed field)
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Label metadata
    PERSONA_NAMES: List[str] = field(default_factory=lambda: [
        "Commentator", "Hero", "Villain", "Guide", "Spectator"
    ])
    EMOTION_NAMES: List[str] = field(default_factory=lambda: [
        "neutral", "excited", "tense", "calm", "triumphant", "fearful"
    ])
    STYLE_NAMES: List[str] = field(default_factory=lambda: [
        "narrative", "intense", "casual", "dramatic"
    ])


# ─── Instantiate Global Config ───────────────────────────────

cfg = NCFNConfig()

# ─── Reproducibility ────────────────────────────────────────

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(cfg.RANDOM_SEED)

# ─── sklearn-free metric helpers ────────────────────────────

def np_train_test_split(arrays, train_frac=0.8, seed=42):
    """Split multiple arrays into train/val with the same random shuffle."""
    rng = np.random.RandomState(seed)
    n = len(arrays[0])
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, val_idx = idx[:split], idx[split:]
    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[val_idx])
    return result

def np_accuracy(y_true, y_pred):
    return float((np.array(y_true) == np.array(y_pred)).mean())

def np_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def np_f1_macro(y_true, y_pred, num_classes):
    cm = np_confusion_matrix(y_true, y_pred, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))

def np_per_class_stats(y_true, y_pred, num_classes):
    cm = np_confusion_matrix(y_true, y_pred, num_classes)
    rows = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        rows.append({"class": c, "precision": prec, "recall": rec, "f1": f1, "support": int(cm[c].sum())})
    return rows

def np_mae(y_true, y_pred):
    return float(np.abs(np.array(y_true) - np.array(y_pred)).mean())

def np_rmse(y_true, y_pred):
    return float(np.sqrt(((np.array(y_true) - np.array(y_pred))**2).mean()))

def np_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))

# ─── Print Summary ───────────────────────────────────────────

print("=" * 60)
print("  Neural Context Fusion Network (NCFN)")
print("  Multimodal AI Voice Generation for Game Streaming")
print("=" * 60)
print(f"\n[Device] Detected: {cfg.DEVICE}")
print(f"[Seaborn] Available: {HAS_SEABORN}")
print(f"\n[Config] NCFNConfig parameters:")
config_dict = asdict(cfg)
for key, value in config_dict.items():
    if not isinstance(value, list):
        print(f"  {key:<28} = {value}")
print("\n[Config] Label names:")
print(f"  PERSONA_NAMES  = {cfg.PERSONA_NAMES}")
print(f"  EMOTION_NAMES  = {cfg.EMOTION_NAMES}")
print(f"  STYLE_NAMES    = {cfg.STYLE_NAMES}")
print("\n[Block 1] Imports, config, and reproducibility — DONE")

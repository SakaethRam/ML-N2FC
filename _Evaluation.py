
# ============================================================
# Block 8 — Evaluation & Metrics
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# --- Load Best Model Weights ---------------------------------

checkpoint = torch.load(BEST_MODEL_PATH, map_location=cfg.DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"[Checkpoint] Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss = {checkpoint['val_loss']:.4f})")

# --- Collect All Val Set Predictions -------------------------

all_persona_pred, all_persona_true = [], []
all_emotion_pred, all_emotion_true = [], []
all_style_pred,   all_style_true   = [], []
all_pitch_pred,   all_pitch_true   = [], []
all_rate_pred,    all_rate_true    = [], []

with torch.no_grad():
    for batch in val_loader:
        speech   = batch["speech"].to(cfg.DEVICE)
        gameplay = batch["gameplay"].to(cfg.DEVICE)
        chat     = batch["chat"].to(cfg.DEVICE)

        preds = model(speech, gameplay, chat)

        all_persona_pred.append(preds["persona_logits"].argmax(dim=1).cpu().numpy())
        all_persona_true.append(batch["persona_label"].numpy())
        all_emotion_pred.append(preds["emotion_logits"].argmax(dim=1).cpu().numpy())
        all_emotion_true.append(batch["emotion_label"].numpy())
        all_style_pred.append(preds["style_logits"].argmax(dim=1).cpu().numpy())
        all_style_true.append(batch["style_label"].numpy())
        all_pitch_pred.append(preds["pitch_pred"].squeeze(-1).cpu().numpy())
        all_pitch_true.append(batch["pitch_shift"].numpy())
        all_rate_pred.append(preds["rate_pred"].squeeze(-1).cpu().numpy())
        all_rate_true.append(batch["speaking_rate"].numpy())

persona_pred = np.concatenate(all_persona_pred)
persona_true = np.concatenate(all_persona_true)
emotion_pred = np.concatenate(all_emotion_pred)
emotion_true = np.concatenate(all_emotion_true)
style_pred   = np.concatenate(all_style_pred)
style_true   = np.concatenate(all_style_true)
pitch_pred   = np.concatenate(all_pitch_pred)
pitch_true   = np.concatenate(all_pitch_true)
rate_pred_v  = np.concatenate(all_rate_pred)
rate_true    = np.concatenate(all_rate_true)

# --- Classification Metrics ----------------------------------

def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    """Compute per-class precision, recall, F1 and macro averages using NumPy helpers."""
    rows = np_per_class_stats(y_true, y_pred, len(class_names))
    df_rows = []
    for r in rows:
        df_rows.append({
            "Class":     class_names[r["class"]],
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1":        round(r["f1"],         4),
            "Support":   r["support"],
        })
    macro_f1 = np_f1_macro(y_true, y_pred, len(class_names))
    df_rows.append({
        "Class":     "macro avg",
        "Precision": round(float(np.mean([r["precision"] for r in rows])), 4),
        "Recall":    round(float(np.mean([r["recall"]    for r in rows])), 4),
        "F1":        round(macro_f1, 4),
        "Support":   len(y_true),
    })
    return pd.DataFrame(df_rows)


persona_df = cls_metrics(persona_true, persona_pred, cfg.PERSONA_NAMES)
emotion_df = cls_metrics(emotion_true, emotion_pred, cfg.EMOTION_NAMES)
style_df   = cls_metrics(style_true,   style_pred,   cfg.STYLE_NAMES)

# --- Regression Metrics --------------------------------------

pitch_mae  = np_mae(pitch_true, pitch_pred)
pitch_rmse = np_rmse(pitch_true, pitch_pred)
pitch_r2   = np_r2(pitch_true, pitch_pred)

rate_mae   = np_mae(rate_true, rate_pred_v)
rate_rmse  = np_rmse(rate_true, rate_pred_v)
rate_r2    = np_r2(rate_true, rate_pred_v)

regression_df = pd.DataFrame([
    {"Task": "Pitch Shift",   "MAE": round(pitch_mae, 4), "RMSE": round(pitch_rmse, 4), "R2": round(pitch_r2, 4)},
    {"Task": "Speaking Rate", "MAE": round(rate_mae, 4),  "RMSE": round(rate_rmse, 4),  "R2": round(rate_r2, 4)},
])

# --- Summary Metrics Dict ------------------------------------

final_metrics = {
    "persona_accuracy": round(np_accuracy(persona_true, persona_pred), 4),
    "persona_macro_f1": round(np_f1_macro(persona_true, persona_pred, cfg.NUM_PERSONAS), 4),
    "emotion_accuracy": round(np_accuracy(emotion_true, emotion_pred), 4),
    "emotion_macro_f1": round(np_f1_macro(emotion_true, emotion_pred, cfg.NUM_EMOTIONS), 4),
    "style_accuracy":   round(np_accuracy(style_true,   style_pred),   4),
    "style_macro_f1":   round(np_f1_macro(style_true,   style_pred,    cfg.NUM_STYLES), 4),
    "pitch_mae":        round(pitch_mae,  4),
    "pitch_rmse":       round(pitch_rmse, 4),
    "pitch_r2":         round(pitch_r2,   4),
    "rate_mae":         round(rate_mae,   4),
    "rate_rmse":        round(rate_rmse,  4),
    "rate_r2":          round(rate_r2,    4),
}

# --- Print Metric Tables -------------------------------------

print("=" * 60)
print("  NCFN Evaluation -- Validation Set")
print("=" * 60)

print("\n-- Persona Head (5 classes) --")
print(persona_df.to_string(index=False))

print("\n-- Emotion Head (6 classes) --")
print(emotion_df.to_string(index=False))

print("\n-- Style Head (4 classes) --")
print(style_df.to_string(index=False))

print("\n-- Regression Heads --")
print(regression_df.to_string(index=False))

print("\n-- Summary Metrics --")
for k, v in final_metrics.items():
    print(f"  {k:<24} = {v}")

# --- Confusion Matrix Grid (2x3) -----------------------------

PALETTE_CMAP = "Blues"

fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
fig_cm.suptitle("NCFN -- Confusion Matrices (Validation Set)", fontweight="bold", fontsize=14)

_tasks = [
    (persona_true, persona_pred, cfg.PERSONA_NAMES, "Persona"),
    (emotion_true, emotion_pred, cfg.EMOTION_NAMES, "Emotion"),
    (style_true,   style_pred,   cfg.STYLE_NAMES,   "Style"),
]

for col, (y_true, y_pred, names, task_name) in enumerate(_tasks):
    cm_counts = np_confusion_matrix(y_true, y_pred, len(names))
    row_sums  = cm_counts.sum(axis=1, keepdims=True)
    cm_norm   = cm_counts.astype(float) / np.where(row_sums == 0, 1, row_sums)

    for row_idx, (data, title_suffix) in enumerate([(cm_counts, "Counts"), (cm_norm, "Normalized")]):
        ax = axes_cm[row_idx, col]
        if HAS_SEABORN:
            import seaborn as sns_lib
            fmt = "d" if row_idx == 0 else ".2f"
            sns_lib.heatmap(
                data, ax=ax, annot=True, fmt=fmt, cmap=PALETTE_CMAP,
                xticklabels=names, yticklabels=names, cbar=True, linewidths=0.3,
            )
        else:
            im = ax.imshow(data, aspect="auto", cmap=PALETTE_CMAP)
            plt.colorbar(im, ax=ax)
            fmt_str = "{:.2f}" if row_idx == 1 else "{:d}"
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    txt = fmt_str.format(val if row_idx == 1 else int(val))
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if val > data.max() * 0.6 else "black")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, fontsize=8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, rotation=0, fontsize=8)
        ax.set_title(f"{task_name} ({title_suffix})", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

fig_eval = fig_cm
plt.close("all")

print("\n[Block 8] Evaluation and confusion matrices -- DONE")

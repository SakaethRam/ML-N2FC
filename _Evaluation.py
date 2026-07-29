
# ============================================================
# Block 8 — Evaluation & Metrics
# ============================================================

# ─── Load Best Model Weights ─────────────────────────────────

checkpoint = torch.load(BEST_MODEL_PATH, map_location=cfg.DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"[Checkpoint] Loaded best model from epoch {checkpoint['epoch']} "
      f"(val_loss = {checkpoint['val_loss']:.4f})")

# ─── Collect All Val Set Predictions ─────────────────────────

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

# ─── Classification Metrics ───────────────────────────────────

def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    """Compute per-class precision, recall, F1 and overall accuracy."""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls_name in class_names:
        if cls_name in report:
            rows.append({
                "Class":    cls_name,
                "Precision": round(report[cls_name]["precision"], 4),
                "Recall":    round(report[cls_name]["recall"],    4),
                "F1":        round(report[cls_name]["f1-score"],   4),
                "Support":   int(report[cls_name]["support"]),
            })
    rows.append({
        "Class":     "macro avg",
        "Precision": round(report["macro avg"]["precision"], 4),
        "Recall":    round(report["macro avg"]["recall"],    4),
        "F1":        round(report["macro avg"]["f1-score"],  4),
        "Support":   int(report["macro avg"]["support"]),
    })
    return pd.DataFrame(rows)


persona_df = cls_metrics(persona_true, persona_pred, cfg.PERSONA_NAMES)
emotion_df = cls_metrics(emotion_true, emotion_pred, cfg.EMOTION_NAMES)
style_df   = cls_metrics(style_true,   style_pred,   cfg.STYLE_NAMES)

# ─── Regression Metrics ───────────────────────────────────────

pitch_mae  = mean_absolute_error(pitch_true, pitch_pred)
pitch_rmse = np.sqrt(mean_squared_error(pitch_true, pitch_pred))
pitch_r2   = r2_score(pitch_true, pitch_pred)

rate_mae   = mean_absolute_error(rate_true, rate_pred_v)
rate_rmse  = np.sqrt(mean_squared_error(rate_true, rate_pred_v))
rate_r2    = r2_score(rate_true, rate_pred_v)

regression_df = pd.DataFrame([
    {"Task": "Pitch Shift",   "MAE": round(pitch_mae, 4), "RMSE": round(pitch_rmse, 4), "R²": round(pitch_r2, 4)},
    {"Task": "Speaking Rate", "MAE": round(rate_mae, 4),  "RMSE": round(rate_rmse, 4),  "R²": round(rate_r2, 4)},
])

# ─── Summary Metrics Dict ─────────────────────────────────────

final_metrics = {
    "persona_accuracy": round(accuracy_score(persona_true, persona_pred), 4),
    "persona_macro_f1": round(f1_score(persona_true, persona_pred, average="macro", zero_division=0), 4),
    "emotion_accuracy": round(accuracy_score(emotion_true, emotion_pred), 4),
    "emotion_macro_f1": round(f1_score(emotion_true, emotion_pred, average="macro", zero_division=0), 4),
    "style_accuracy":   round(accuracy_score(style_true,   style_pred),   4),
    "style_macro_f1":   round(f1_score(style_true, style_pred, average="macro", zero_division=0), 4),
    "pitch_mae":        round(pitch_mae,  4),
    "pitch_rmse":       round(pitch_rmse, 4),
    "pitch_r2":         round(pitch_r2,   4),
    "rate_mae":         round(rate_mae,   4),
    "rate_rmse":        round(rate_rmse,  4),
    "rate_r2":          round(rate_r2,    4),
}

# ─── Print Metric Tables ──────────────────────────────────────

print("=" * 60)
print("  NCFN Evaluation — Validation Set")
print("=" * 60)

print("\n── Persona Head (5 classes) ──")
print(persona_df.to_string(index=False))

print("\n── Emotion Head (6 classes) ──")
print(emotion_df.to_string(index=False))

print("\n── Style Head (4 classes) ──")
print(style_df.to_string(index=False))

print("\n── Regression Heads ──")
print(regression_df.to_string(index=False))

print("\n── Summary Metrics ──")
for k, v in final_metrics.items():
    print(f"  {k:<24} = {v}")

# ─── Confusion Matrix Grid (2×3) ─────────────────────────────

PALETTE_CMAP = "Blues"

fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
fig_cm.suptitle("NCFN — Confusion Matrices (Validation Set)", fontweight="bold", fontsize=14)

_tasks = [
    (persona_true, persona_pred, cfg.PERSONA_NAMES, "Persona"),
    (emotion_true, emotion_pred, cfg.EMOTION_NAMES, "Emotion"),
    (style_true,   style_pred,   cfg.STYLE_NAMES,   "Style"),
]

for col, (y_true, y_pred, names, task_name) in enumerate(_tasks):
    cm     = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    for row, (data, title_suffix) in enumerate([(cm, "Counts"), (cm_norm, "Normalized")]):
        ax = axes_cm[row, col]
        fmt = "d" if row == 0 else ".2f"
        sns.heatmap(
            data,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap=PALETTE_CMAP,
            xticklabels=names,
            yticklabels=names,
            cbar=True,
            linewidths=0.3,
        )
        ax.set_title(f"{task_name} ({title_suffix})", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

fig_eval = fig_cm
plt.close("all")

print("\n[Block 8] Evaluation and confusion matrices — DONE")

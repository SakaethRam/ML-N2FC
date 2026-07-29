
# ============================================================
# Block 9 — Training Visualizations
# ============================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

VISUALIZATION_PATH = "./ncfn_training_results.png"
VIZ_DPI            = 200
SCATTER_SAMPLE_N   = 200

# Color palette (dark background, elegant muted tones)
DARK_BG     = "#1a1a2e"
DARK_AX     = "#16213e"
ACCENT_BLUE = "#4CC9F0"
ACCENT_PINK = "#F72585"
ACCENT_GRN  = "#7BF1A8"
ACCENT_YEL  = "#F8C537"
ACCENT_PUR  = "#A55FE3"
ACCENT_ORNG = "#FF6B35"
GRID_COL    = "#2e2e50"
TEXT_COL    = "#e0e0e0"

plt.rcParams.update({
    "text.color":        TEXT_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_AX,
    "axes.edgecolor":    "#444466",
    "axes.grid":         True,
    "grid.color":        GRID_COL,
    "grid.alpha":        0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlecolor":   TEXT_COL,
})

epochs_x = list(range(1, cfg.NUM_EPOCHS + 1))

fig = plt.figure(figsize=(20, 18), constrained_layout=True, dpi=VIZ_DPI)
fig.set_facecolor(DARK_BG)
fig.suptitle(
    "Neural Context Fusion Network -- Training Results",
    fontweight="bold", fontsize=16, color=TEXT_COL
)

gs = gridspec.GridSpec(3, 2, figure=fig)

# Panel 1: Train vs Val Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(DARK_AX)
ax1.plot(epochs_x, history["train_loss"], color=ACCENT_BLUE, lw=2, label="Train Loss")
ax1.plot(epochs_x, history["val_loss"],   color=ACCENT_PINK, lw=2, label="Val Loss", linestyle="--")
ax1.set_title("a. Train vs Validation Loss", fontsize=11, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend(frameon=False, labelcolor=TEXT_COL)

best_epoch_idx = int(np.argmin(history["val_loss"]))
best_val_v     = history["val_loss"][best_epoch_idx]
_val_range     = max(history["val_loss"]) - min(history["val_loss"])
ax1.annotate(
    f"Best: {best_val_v:.3f}\n(ep {best_epoch_idx + 1})",
    xy=(best_epoch_idx + 1, best_val_v),
    xytext=(min(best_epoch_idx + 4, cfg.NUM_EPOCHS - 2), best_val_v + _val_range * 0.25),
    fontsize=8, color=ACCENT_PINK,
    arrowprops=dict(arrowstyle="->", color=ACCENT_PINK, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc=DARK_AX, ec=ACCENT_PINK, alpha=0.7),
)

# Panel 2: Val Classification Accuracies
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(DARK_AX)
ax2.plot(epochs_x, history["val_persona_acc"], color=ACCENT_BLUE, lw=2, label="Persona Acc")
ax2.plot(epochs_x, history["val_emotion_acc"], color=ACCENT_YEL,  lw=2, label="Emotion Acc", linestyle="--")
ax2.plot(epochs_x, history["val_style_acc"],   color=ACCENT_PUR,  lw=2, label="Style Acc",   linestyle=":")
ax2.set_title("b. Validation Classification Accuracies", fontsize=11, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.05)
ax2.legend(frameon=False, labelcolor=TEXT_COL)

# Panel 3: Val Pitch MAE
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(DARK_AX)
ax3.plot(epochs_x, history["val_pitch_mae"], color=ACCENT_ORNG, lw=2)
ax3.set_title("c. Validation Pitch MAE (semitones)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("MAE (semitones)")
ax3.axhline(
    y=min(history["val_pitch_mae"]),
    color=ACCENT_ORNG, linestyle=":", lw=1.2, alpha=0.7,
    label=f"Min: {min(history['val_pitch_mae']):.3f}"
)
ax3.legend(frameon=False, labelcolor=TEXT_COL)

# Panel 4: Val Speaking Rate MAE
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(DARK_AX)
ax4.plot(epochs_x, history["val_rate_mae"], color=ACCENT_PINK, lw=2)
ax4.set_title("d. Validation Speaking Rate MAE", fontsize=11, fontweight="bold")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("MAE (rate multiplier)")
ax4.axhline(
    y=min(history["val_rate_mae"]),
    color=ACCENT_PINK, linestyle=":", lw=1.2, alpha=0.7,
    label=f"Min: {min(history['val_rate_mae']):.3f}"
)
ax4.legend(frameon=False, labelcolor=TEXT_COL)

# Panel 5: Bar Chart -- Final Val Accuracy per Head
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor(DARK_AX)

clf_tasks  = ["Persona", "Emotion", "Style"]
clf_accs   = [
    history["val_persona_acc"][-1],
    history["val_emotion_acc"][-1],
    history["val_style_acc"][-1],
]
bar_colors = [ACCENT_BLUE, ACCENT_YEL, ACCENT_PUR]
bars = ax5.bar(clf_tasks, clf_accs, color=bar_colors, width=0.55, edgecolor=DARK_AX, linewidth=0.5)

for bar, acc in zip(bars, clf_accs):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center", va="bottom", fontsize=10, color=TEXT_COL,
    )

ax5.set_title("e. Final Validation Accuracy -- Classification Heads", fontsize=11, fontweight="bold")
ax5.set_ylabel("Accuracy")
ax5.set_ylim(0, 1.15)

# Panel 6: Scatter -- Predicted vs Actual Pitch Shift
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_facecolor(DARK_AX)

_rng = np.random.RandomState(42)
_idx = _rng.choice(len(pitch_true), min(SCATTER_SAMPLE_N, len(pitch_true)), replace=False)
_pt  = pitch_true[_idx]
_pp  = pitch_pred[_idx]

ax6.scatter(_pt, _pp, alpha=0.55, s=20, color=ACCENT_GRN, edgecolors="none")
_lim = max(abs(_pt.min()), abs(_pt.max())) * 1.1
ax6.plot([-_lim, _lim], [-_lim, _lim], color=ACCENT_YEL, lw=1.5, linestyle="--", label="Perfect fit")
ax6.set_title(f"f. Predicted vs Actual Pitch Shift (val, n={SCATTER_SAMPLE_N})", fontsize=11, fontweight="bold")
ax6.set_xlabel("Actual Pitch Shift (semitones)")
ax6.set_ylabel("Predicted Pitch Shift (semitones)")
ax6.legend(frameon=False, labelcolor=TEXT_COL)
ax6.text(
    0.05, 0.92,
    f"MAE={pitch_mae:.3f}  R2={pitch_r2:.3f}",
    transform=ax6.transAxes,
    fontsize=9, color=TEXT_COL,
    bbox=dict(boxstyle="round,pad=0.3", fc=DARK_AX, ec="#444466", alpha=0.8),
)

# Save & Display
fig.savefig(VISUALIZATION_PATH, dpi=VIZ_DPI, bbox_inches="tight", facecolor=DARK_BG)
fig_training = fig
plt.close("all")

# Reset rcParams to defaults for downstream blocks
plt.rcParams.update(plt.rcParamsDefault)

print("=" * 60)
print("  NCFN Training Visualizations")
print("=" * 60)
print(f"\n[Saved] {VISUALIZATION_PATH}")
print("[Panels]")
print("  a. Train vs Val Loss curve")
print("  b. Val classification accuracies (persona, emotion, style)")
print("  c. Val pitch MAE over epochs")
print("  d. Val speaking rate MAE over epochs")
print("  e. Final accuracy bar chart (classification heads)")
print(f"  f. Predicted vs actual pitch scatter (n={SCATTER_SAMPLE_N})")
print("\n[Block 9] Training visualizations -- DONE")

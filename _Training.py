
# ============================================================
# Block 7 — Multi-Task Loss & Training Loop
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import asdict
from typing import Dict, Tuple
from torch.utils.data import DataLoader

BEST_MODEL_PATH    = "./ncfn_best_model.pt"
LOG_EVERY_N_EPOCHS = 5


# --- Multi-Task Uncertainty-Weighted Loss (Kendall et al. 2018) ---

class MultiTaskLoss(nn.Module):
    """
    Learns per-task log-variance (log_var) parameters to automatically
    balance classification and regression losses.

    For classification tasks:
        L_i = (1 / exp(log_var_i)) * CE_i  +  0.5 * log_var_i

    For regression tasks:
        L_i = (1 / (2 * exp(log_var_i))) * MSE_i  +  0.5 * log_var_i
    """

    TASK_NAMES  = ["persona", "emotion", "pitch", "rate", "style"]
    TASK_TYPES  = ["cls",     "cls",     "reg",   "reg",  "cls"]   # cls or reg

    def __init__(self):
        super().__init__()
        # One learnable log-variance per task; initialized near 0
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in self.TASK_NAMES
        ])

        self.ce_loss  = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets:     Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns total weighted loss and a dict of per-task unweighted losses.
        """
        # Per-task raw losses
        l_persona = self.ce_loss(
            predictions["persona_logits"], targets["persona_label"]
        )
        l_emotion = self.ce_loss(
            predictions["emotion_logits"], targets["emotion_label"]
        )
        l_pitch = self.mse_loss(
            predictions["pitch_pred"].squeeze(-1), targets["pitch_shift"]
        )
        l_rate = self.mse_loss(
            predictions["rate_pred"].squeeze(-1), targets["speaking_rate"]
        )
        l_style = self.ce_loss(
            predictions["style_logits"], targets["style_label"]
        )

        raw_losses = [l_persona, l_emotion, l_pitch, l_rate, l_style]
        task_losses = {
            name: raw.item()
            for name, raw in zip(self.TASK_NAMES, raw_losses)
        }

        # Uncertainty-weighted combination
        total = torch.zeros(1, device=next(self.parameters()).device)

        for i, (raw, task_type) in enumerate(zip(raw_losses, self.TASK_TYPES)):
            log_var = self.log_vars[i]
            if task_type == "cls":
                total = total + torch.exp(-log_var) * raw + 0.5 * log_var
            else:
                total = total + 0.5 * torch.exp(-log_var) * raw + 0.5 * log_var

        return total.squeeze(), task_losses


# --- Optimizer & Scheduler -----------------------------------

multi_task_loss_fn = MultiTaskLoss().to(cfg.DEVICE)

optimizer = optim.Adam(
    list(model.parameters()) + list(multi_task_loss_fn.parameters()),
    lr=cfg.LEARNING_RATE,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg.NUM_EPOCHS,
    eta_min=cfg.LEARNING_RATE * 0.01,
)


# --- Metric Tracking -----------------------------------------

history = {
    "train_loss":      [],
    "val_loss":        [],
    "val_persona_acc": [],
    "val_emotion_acc": [],
    "val_style_acc":   [],
    "val_pitch_mae":   [],
    "val_rate_mae":    [],
}


# --- Helper: Evaluation Pass ---------------------------------

def evaluate(loader: DataLoader) -> Dict[str, float]:
    """Run model on loader, return aggregated metrics."""
    model.eval()
    multi_task_loss_fn.eval()

    total_loss  = 0.0
    n_batches   = 0

    all_persona_pred,  all_persona_true  = [], []
    all_emotion_pred,  all_emotion_true  = [], []
    all_style_pred,    all_style_true    = [], []
    all_pitch_pred,    all_pitch_true    = [], []
    all_rate_pred,     all_rate_true     = [], []

    with torch.no_grad():
        for batch in loader:
            speech   = batch["speech"].to(cfg.DEVICE)
            gameplay = batch["gameplay"].to(cfg.DEVICE)
            chat     = batch["chat"].to(cfg.DEVICE)

            targets = {
                "persona_label": batch["persona_label"].to(cfg.DEVICE),
                "emotion_label": batch["emotion_label"].to(cfg.DEVICE),
                "pitch_shift":   batch["pitch_shift"].to(cfg.DEVICE),
                "speaking_rate": batch["speaking_rate"].to(cfg.DEVICE),
                "style_label":   batch["style_label"].to(cfg.DEVICE),
            }

            preds = model(speech, gameplay, chat)
            loss, _ = multi_task_loss_fn(preds, targets)
            total_loss += loss.item()
            n_batches  += 1

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

    return {
        "val_loss":        total_loss / max(n_batches, 1),
        "val_persona_acc": np_accuracy(persona_true, persona_pred),
        "val_emotion_acc": np_accuracy(emotion_true, emotion_pred),
        "val_style_acc":   np_accuracy(style_true, style_pred),
        "val_pitch_mae":   np_mae(pitch_true, pitch_pred),
        "val_rate_mae":    np_mae(rate_true, rate_pred_v),
    }


# --- Training Loop -------------------------------------------

best_val_loss  = float("inf")
train_start    = time.time()

print("=" * 60)
print("  NCFN Training")
print(f"  Epochs: {cfg.NUM_EPOCHS}  |  LR: {cfg.LEARNING_RATE}  |  "
      f"Batch: {cfg.BATCH_SIZE}  |  Device: {cfg.DEVICE}")
print("=" * 60)

for epoch in range(1, cfg.NUM_EPOCHS + 1):
    # Train Phase
    model.train()
    multi_task_loss_fn.train()
    epoch_train_loss = 0.0
    n_train_batches  = 0

    for batch in train_loader:
        speech   = batch["speech"].to(cfg.DEVICE)
        gameplay = batch["gameplay"].to(cfg.DEVICE)
        chat     = batch["chat"].to(cfg.DEVICE)

        targets = {
            "persona_label": batch["persona_label"].to(cfg.DEVICE),
            "emotion_label": batch["emotion_label"].to(cfg.DEVICE),
            "pitch_shift":   batch["pitch_shift"].to(cfg.DEVICE),
            "speaking_rate": batch["speaking_rate"].to(cfg.DEVICE),
            "style_label":   batch["style_label"].to(cfg.DEVICE),
        }

        optimizer.zero_grad()
        preds = model(speech, gameplay, chat)
        loss, _task_losses = multi_task_loss_fn(preds, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(multi_task_loss_fn.parameters()),
            max_norm=1.0
        )

        optimizer.step()
        epoch_train_loss += loss.item()
        n_train_batches  += 1

    scheduler.step()

    avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
    history["train_loss"].append(avg_train_loss)

    # Validation Phase
    val_metrics = evaluate(val_loader)
    for k, v in val_metrics.items():
        history[k].append(v)

    # Save Best Model
    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "loss_state":   multi_task_loss_fn.state_dict(),
            "val_loss":     best_val_loss,
            "config":       asdict(cfg),
        }, BEST_MODEL_PATH)

    # Logging
    if epoch % LOG_EVERY_N_EPOCHS == 0 or epoch == 1:
        elapsed = time.time() - train_start
        print(
            f"Epoch {epoch:>3}/{cfg.NUM_EPOCHS}  |  "
            f"Train Loss: {avg_train_loss:.4f}  |  "
            f"Val Loss: {val_metrics['val_loss']:.4f}  |  "
            f"P-Acc: {val_metrics['val_persona_acc']:.3f}  |  "
            f"E-Acc: {val_metrics['val_emotion_acc']:.3f}  |  "
            f"S-Acc: {val_metrics['val_style_acc']:.3f}  |  "
            f"Pitch MAE: {val_metrics['val_pitch_mae']:.3f}  |  "
            f"Rate MAE: {val_metrics['val_rate_mae']:.3f}  |  "
            f"[{elapsed:.0f}s]"
        )

total_time = time.time() - train_start

print("=" * 60)
print(f"  Training Complete in {total_time:.1f}s")
print(f"  Best Val Loss: {best_val_loss:.4f}")
print(f"  Model saved -> {BEST_MODEL_PATH}")
print("=" * 60)
print("\n[Block 7] Multi-task training loop -- DONE")

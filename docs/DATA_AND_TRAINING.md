# Data and Training

## Synthetic data generation (Block 02)

5,000 samples are generated covering all three modalities:

- Speech embeddings: 256-dim
- Gameplay telemetry: padded to 128-dim
- Chat features: padded to 64-dim

Labels for persona, emotion, style, pitch shift, and speaking rate are generated with intentional correlations to the input features (see `MODEL_DESIGN_DECISIONS.md`, decision 5), not independently at random. This is synthetic data standing in for real streaming data that doesn't yet exist in this repository; treat the 5,000-sample scale and the specific correlation rules as a development/prototyping baseline, not a claim about what a production-scale, real-data training run would look like.

## Dataset and loaders (Block 03)

- `NCFNDataset`: a PyTorch `Dataset` wrapping the three modality tensors plus the five label sets.
- An 80/20 **temporal** split for train/validation, not a random shuffle-split. A temporal split matters specifically for streaming data: it evaluates the model on data that comes chronologically after what it trained on, which is a more honest test of "would this generalize to the next stream" than a random split would be, since a random split can leak near-identical nearby-in-time samples into both train and validation.
- `DataLoader` construction with shuffling applied at batch level (standard for training), on top of the temporal split at the dataset level.

## Training (Block 07)

- Multi-task training loop, with the uncertainty-weighted loss described in `MODEL_DESIGN_DECISIONS.md`.
- Adam optimizer with a learning-rate scheduler.
- Best-checkpoint saving: the training loop tracks and persists the best-performing checkpoint rather than only the final epoch's weights, which matters for a multi-task setup where the final epoch isn't guaranteed to be the best epoch across all five heads simultaneously.
- Per-epoch metrics logging, using the NumPy-only metric functions from Block 01.

## What isn't specified in the README, worth confirming in source before relying on it

- The exact learning-rate schedule (step size, decay factor, or whether it's a plateau-based scheduler) isn't documented at the README level.
- The checkpoint-selection criterion (which metric determines "best") across five heterogeneous heads isn't specified; confirm whether it's a single head's metric, an average across heads, or something else in Block 07's actual implementation before assuming which checkpoint gets kept during a long training run.
- Batch size is stated as `BATCH_SIZE=64` in the hyperparameters (Block 01), but total training epoch count is not stated in the README; check Block 07 directly.

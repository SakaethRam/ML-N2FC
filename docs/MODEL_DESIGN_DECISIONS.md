# Model Design Decisions

This expands on the six design decisions the README calls out, with the reasoning behind each spelled out rather than just stated.

## 1. Cross-modal attention fusion over concatenation

Covered in depth in `ARCHITECTURE.md`. The short version: concatenation is static, attention is data-dependent. Given this is meant to work over a live, rapidly-changing game stream, a static weighting between "what the player is doing" and "what chat is saying" would miss exactly the moments (a sudden clutch play, a chat pile-on) where the right modality to weight most heavily changes turn to turn.

## 2. Multi-task learning with uncertainty-weighted loss

Five prediction heads means five loss terms (three classification, two regression) that are not naturally on the same scale: cross-entropy loss and MAE/MSE-style regression loss don't share units, and hand-tuning five scale factors is fragile and has to be redone every time a head's target distribution changes. Kendall uncertainty weighting sidesteps this by giving each head a learnable log-variance parameter, which the training loop uses to automatically downweight a head that's producing noisy/overconfident predictions and upweight one that's under-contributing. The practical benefit: adding a sixth head later wouldn't require re-tuning the other five loss weights by hand.

## 3. Modality-specific encoders instead of a shared encoder

Also covered in `ARCHITECTURE.md`. Worth adding here: this is a common pattern in multimodal learning generally (see also CLIP-style dual encoders), and the reasoning transfers directly: fusing too early, before each modality has had a chance to build its own useful representation, tends to let the highest-magnitude or easiest-to-fit modality dominate the shared representation. Separate encoders with attention gating let each modality "earn" its contribution to the fused representation on its own terms first.

## 4. NumPy-only evaluation metrics, no scikit-learn

`np_accuracy`, `np_f1_macro`, `np_confusion_matrix`, `np_per_class_stats`, `np_mae`, `np_rmse`, `np_r2` are all implemented directly in Block 01, deliberately avoiding a scikit-learn dependency. Two practical reasons this matters beyond "one fewer pip install":

- **Deployment footprint** — if NCFN or its evaluation code ever needs to run in a latency-sensitive or dependency-constrained environment (e.g. alongside the live inference pipeline itself), not needing scikit-learn keeps the dependency surface smaller.
- **Determinism and transparency** — hand-rolled metric implementations mean there's no version-to-version behavior drift from an external library's own metric implementation changing; what the metric computes is fully visible in this codebase rather than delegated to a black-box import.

The README states these produce identical results to their sklearn counterparts; if you extend the evaluation metrics, it's worth keeping a small parity test against sklearn's implementations during development even though the shipped code doesn't depend on sklearn at runtime.

## 5. Synthetic data with correlated labels, not independent random labels

Random, uncorrelated synthetic labels would let the model "learn" the training data with the right architecture but would teach it nothing about the actual relationships it needs in production, since there'd be no real signal to find. Instead, the synthetic generator builds in intentional correlations: gameplay intensity drives emotion and pitch variation, health state influences persona assignment, chat sentiment modulates expressive style. This means the model is learning genuine cross-modal dependencies during training, even on synthetic data, which should transfer more usefully once real streaming data becomes available than a model trained on uncorrelated random labels would.

## 6. `FUSION_DIM=768`

Covered in `ARCHITECTURE.md`. Restated briefly here because it's as much a design decision as an architectural fact: 768 was chosen specifically to keep both the modality-split (÷3) and the attention-head-split (÷8) landing on integers, avoiding shape errors that a less deliberately chosen dimension could introduce.

## What to check before extending this design

- If you add a sixth prediction head, confirm the uncertainty-weighting implementation in Block 07 actually generalizes to N heads rather than being hardcoded for five.
- If you change `NUM_ATTENTION_HEADS`, re-verify the `FUSION_DIM` divisibility logic from decision 6 still holds; it's not automatically re-derived.
- If you eventually train on real streaming data instead of synthetic, the correlated-label design in decision 5 won't apply directly; real labels would need their own collection/annotation pipeline, which this repository does not currently include.

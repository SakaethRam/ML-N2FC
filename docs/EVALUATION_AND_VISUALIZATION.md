# Evaluation and Visualization

## Evaluation (Block 08)

Full validation-set evaluation, per head:

- **Classification heads (Persona, Emotion, Style):** accuracy and macro F1, plus per-class breakdowns and confusion matrices.
- **Regression heads (Pitch Shift, Speaking Rate):** MAE, RMSE, R².

All computed with the NumPy-only metric functions from Block 01 (see `MODEL_DESIGN_DECISIONS.md`, decision 4), not scikit-learn.

### Why macro F1, not accuracy alone, for the classification heads

Persona (5 classes) and Emotion (7 classes) are both multi-class with no guarantee of balanced class frequency in the synthetic (or eventually real) data. Accuracy alone can look good on an imbalanced dataset by mostly predicting the majority class; macro F1 weights every class equally regardless of its frequency, which is the more honest metric for whether the model is actually distinguishing all five personas or all seven emotions, not just the most common one.

## Visualization (Block 09)

A six-panel training dashboard:

1. Loss curves (training vs. validation, across epochs)
2. Accuracy curves (for the classification heads)
3. Regression MAE curves (for pitch and rate)
4. Final accuracy bar chart (per classification head)
5. Pitch scatter plot (predicted vs. actual)
6. Speaking rate scatter plot (predicted vs. actual)

The two scatter plots (5 and 6) are worth paying particular attention to during development: a regression head can post a reasonable aggregate MAE/RMSE while still showing systematic bias (e.g. consistently under-predicting pitch shift at the extremes) that only becomes visible in a predicted-vs-actual scatter, not in a single aggregate number.

## Reading the dashboard as a diagnostic, not just a report

- If loss curves diverge (training loss falling, validation loss rising), that's the standard overfitting signal, more visible here than in the aggregate metrics from Block 08 alone.
- If one classification head's accuracy curve plateaus much earlier than the others, that's a candidate sign the uncertainty-weighting from `MODEL_DESIGN_DECISIONS.md` (decision 2) may need inspecting for that head specifically, since it should in principle equalize how much attention each head gets during training.
- The scatter plots are the most direct way to catch a regression head that's technically hitting a decent aggregate score while being systematically wrong in a specific, correctable way (e.g. always predicting near the midpoint of the range rather than the extremes).

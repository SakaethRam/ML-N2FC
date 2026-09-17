# Contributing

## Before you start

- This codebase runs as sequential blocks in a shared namespace (see `docs/SETUP_AND_USAGE.md`), not as an imported package. A change to an earlier block (e.g. adding a field to `NCFNDataset` in Block 03) can silently affect every later block that depends on it existing in the shared namespace, without Python's import system catching a missing-name error until runtime. Test the full `run_pipeline.py` sequence after any change to an earlier block, not just the block you directly edited.
- Changes to `FUSION_DIM` or `NUM_ATTENTION_HEADS` (Block 01) should be checked against the divisibility reasoning in `docs/MODEL_DESIGN_DECISIONS.md` (decision 6) before being committed; a value that breaks the ÷3 / ÷8 divisibility will cause shape errors in the fusion layer.
- Changes to the multi-task loss weighting (Block 07) should include before/after per-head metrics from Block 08's evaluation, since a change here affects all five heads simultaneously, not just one.

## Workflow

1. Fork the repository.
2. Create a feature branch.
3. Implement your change, scoped to as few blocks as possible.
4. Run `python run_pipeline.py` end to end and confirm Block 08's evaluation output and Block 09's dashboard both still look reasonable, not just that the script exits without error.
5. Submit a pull request with a clear description of what changed, which block(s) it touches, and any metric shifts observed in evaluation.

## Where to make changes

| Area | Block / File |
|------|--------------|
| Hyperparameters, custom metrics | Block 01, `Imports&Config.py` |
| Synthetic data generation | Block 02, `Data-Generation.py` |
| Dataset / DataLoader | Block 03, `Dataset&Loaders.py` |
| Modality encoders | Block 04, `_Encoders.py` |
| Fusion layer | Block 05, `Fusion-Layer.py` |
| Full model definition | Block 06, `NCFN-Model.py` |
| Training loop | Block 07, `_Training.py` |
| Evaluation metrics | Block 08, `_Evaluation.py` |
| Visualization dashboard | Block 09, `_Visualization.py` |
| Inference / ElevenLabs mapping | Block 10, `Inference-Pipeline.py` |
| Runner script | `run_pipeline.py` (this delivery) |
| CI | `.github/workflows/` |

## Reporting issues

Use the repository's Issues tab for bugs, training instability, or documentation gaps. If a bug only reproduces after several blocks run in sequence (which, given the shared-namespace design, is a real category of bug here), include which block introduced the problem and which later block first exhibits it, not just the final error message.

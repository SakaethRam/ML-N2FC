# Setup and Usage

## Important: how this codebase is actually meant to run

The README documents this repository as 10 numbered "blocks" (Block 01 through Block 10), and several of the corresponding filenames contain characters that are **not valid in a Python `import` statement**: hyphens (`Fusion-Layer.py`, `NCFN-Model.py`, `Inference-Pipeline.py`, `Data-Generation.py`) and an ampersand (`Imports&Config.py`, `Dataset&Loaders.py`). `import` requires a valid Python identifier, and none of these filenames qualify as one.

This strongly indicates the code was originally written and run as notebook cells (each block a cell), later exported as individual `.py` files preserving the cell boundaries, rather than designed as an importable package from the start. The README has no Installation or Usage section, which is consistent with this: running it was presumably always "open in a notebook, run cells 1 through 10 in order," not "pip install and import."

**Practical implication:** you cannot `import` these files directly by their current names. Your options:

1. **Run them in a Jupyter notebook**, pasting each block's content into a cell in order (01 → 10), which most closely matches how this was likely developed.
2. **Use the `exec()`-based runner provided in this delivery** (`run_pipeline.py`), which reads and executes each block's source in a shared namespace, in the documented order, without needing to rename anything in the original repository.
3. **Rename the files to valid module names** (e.g. `01_imports_and_config.py`, `02_data_generation.py`, ...) if you want a real importable package going forward. This is a bigger, more deliberate change to the repository's structure and isn't done automatically as part of this delivery; see "Optional: converting to a proper package" below if you want to go this route.

## Running via the provided runner script

```bash
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` executes the 10 blocks in the README's documented order, in one shared namespace, so that later blocks (which depend on classes/functions/variables defined in earlier blocks, e.g. `NCFNDataset` from Block 03 or the model class from Block 06) see everything earlier blocks defined, exactly as they would inside a single notebook session.

## Dependencies

Per the README, stated explicitly as the complete list (scikit-learn intentionally excluded, see `MODEL_DESIGN_DECISIONS.md`):

```
torch>=2.0
numpy
pandas
matplotlib
```

Python 3.9+ is required.

## Optional: converting to a proper package

If you want NCFN to be pip-installable or importable elsewhere (e.g. from the inference pipeline as a library rather than a script), the files would need renaming to valid Python module names and their cross-block references (currently implicit, via shared notebook/exec namespace) made explicit via proper `import` statements between modules. This is a meaningful refactor, not a rename-only change, since code in later blocks currently relies on names simply existing in the shared global namespace rather than being explicitly imported. Treat this as a roadmap item rather than something to do casually alongside an unrelated change.

## Full setup checklist

- [ ] Python 3.9+ available
- [ ] `pip install -r requirements.txt`
- [ ] Either: run blocks in a notebook in order, or `python run_pipeline.py`
- [ ] Confirm training completes and a checkpoint is saved (Block 07) before relying on Block 10's inference pipeline, since it depends on trained weights
- [ ] For live ElevenLabs integration, complete the account-specific setup in `docs/INFERENCE_AND_ELEVENLABS.md` first

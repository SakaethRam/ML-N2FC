"""
run_pipeline.py

Runs NCFN's 10 code blocks in the order documented in the README, in
one shared namespace, matching how the project appears to have been
developed (as notebook cells) and run. See docs/SETUP_AND_USAGE.md
for why this exists: several of the block filenames (containing "-"
or "&") are not valid Python module names and cannot be `import`-ed
directly.

Usage:
    python run_pipeline.py

This does not rename or modify any file in the original repository;
it reads each block's source and executes it via `exec`, in sequence,
so that names defined in an earlier block (classes, functions,
config constants) are visible to every later block, exactly as they
would be across cells in a single notebook session.
"""

from __future__ import annotations

import pathlib
import sys

# Order matches the README's "Codebase Structure" table (Block 01 - 10).
BLOCKS = [
    "Imports&Config.py",
    "Data-Generation.py",
    "Dataset&Loaders.py",
    "_Encoders.py",
    "Fusion-Layer.py",
    "NCFN-Model.py",
    "_Training.py",
    "_Evaluation.py",
    "_Visualization.py",
    "Inference-Pipeline.py",
]


def run_pipeline(repo_root: str | pathlib.Path = ".") -> dict:
    """
    Execute all 10 blocks in order within one shared namespace.
    Returns that namespace, so a caller (or an interactive session)
    can inspect trained models, dataloaders, or results afterward
    rather than only running this as a fire-and-forget script.
    """
    root = pathlib.Path(repo_root)
    namespace: dict = {"__name__": "__main__"}

    for block_name in BLOCKS:
        block_path = root / block_name
        if not block_path.exists():
            print(f"!! Missing block file: {block_path}", file=sys.stderr)
            print(
                "   Place all 10 block files at the repository root "
                "(or pass the correct --root) before running.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"--- Running {block_name} ---")
        source = block_path.read_text(encoding="utf-8")
        code = compile(source, filename=str(block_path), mode="exec")
        exec(code, namespace)  # noqa: S102 - intentional, see module docstring

    return namespace


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="Directory containing the 10 block .py files (default: current directory).",
    )
    args = parser.parse_args()

    run_pipeline(args.root)
    print("--- Pipeline complete ---")

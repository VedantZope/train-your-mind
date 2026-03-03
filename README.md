MIND: Medical Image Nonrigid Deformation
=======================================

This repository contains code for evaluating nonrigid registration methods on the LungCT Learn2Reg dataset using FireANTs and EvoTorch-based models.

### Inference / Evaluation Scripts

Inference is driven by the module entrypoint `mind.evotorch.infer.main`, which is exposed as a thin wrapper in `scripts/infer.py`.

- **`scripts/infer.py`**: CLI entrypoint for running inference / evaluation.
  - Simply imports `main` from `mind.evotorch.infer` and executes it.
  - Use this script when you want a straightforward command-line interface without touching the library code.

- **`src/mind/evotorch/infer.py`**: Core implementation of inference and evaluation.
  - Builds fixed/moving image pairs and associated keypoints and (optionally) masks.
  - Runs greedy registration using FireANTs, optionally with:
    - **identity** features (raw images),
    - **MIND** descriptor features,
    - or a learned **model** loaded from checkpoint weights.
  - Computes mean Target Registration Error (TRE) across all evaluated pairs.
  - Can optionally save displacement fields in Learn2Reg format.

#### Expected Data Layout

By default, inference expects a preprocessed LungCT dataset under:

- **Default root**: `/data/vedant/MIND/data/LungCT_L2R_preprocessed`

Within this root, the following directories must exist:

- **`imagesTr/`** and **`imagesTs/`**: fixed/moving CT volumes for train/test splits.
- **`keypointsTr/`** and **`keypointsTs/`**: CSV keypoints for train/test splits.
- **`masksTr/`** and **`masksTs/`** (optional, required when using masked CC loss): binary masks aligned with the images.

You can override the dataset root via the `--data-root` flag.

#### Command-Line Interface

The main CLI is defined in `mind.evotorch.infer.main` and is exposed via `scripts/infer.py`:

- **Required split selection (choose exactly one)**:
  - `--train`: evaluate on the train split.
  - `--test`: evaluate on the test split.

- **Masked loss control (choose at most one)**:
  - `--masked`: use masked CC loss (requires masks directories to exist).
  - `--no-masked`: disable masked loss and use standard CC.

- **Model selection** (`--model`):
  - `auto` (default): equivalent to `model` (uses the checkpoint specified by `--weights`).
  - `model`: use a learned feature extractor loaded from a `.pt` checkpoint.
  - `mind`: use MIND descriptor features (no weights required).
  - `identity`: use the raw image intensities as features.

- **Weights**:
  - `--weights PATH`: path to a model checkpoint (`.pt`), **required** when `--model auto` or `--model model` is used.
  - The checkpoint directory is expected to contain a `model_config.json` file next to (or one directory above) the weights file, which is used to rebuild the model architecture.

- **Dataset root**:
  - `--data-root PATH`: overrides the default dataset root directory.

- **Displacement saving**:
  - `--save-displacement`: save displacement fields (warps) in Learn2Reg-compatible format using `reg.save_as_scipy_transforms`.
  - Results are written under `infer/results/` inside the project (see `results_dir` in `run_inference`).

#### Example Commands

- **Evaluate a learned model on the test split with masked CC**:

```bash
python scripts/infer.py \
  --test \
  --masked \
  --model auto \
  --weights /path/to/checkpoints/model_best.pt \
  --data-root /data/vedant/MIND/data/LungCT_L2R_preprocessed
```

- **Evaluate using MIND descriptors on the train split without masks**:

```bash
python scripts/infer.py \
  --train \
  --no-masked \
  --model mind \
  --data-root /data/vedant/MIND/data/LungCT_L2R_preprocessed
```

- **Evaluate simple identity (raw intensity) registration on the test split**:

```bash
python scripts/infer.py \
  --test \
  --no-masked \
  --model identity \
  --data-root /data/vedant/MIND/data/LungCT_L2R_preprocessed
```

Each run prints the mean TRE for the chosen split:

- **Output**: a line of the form `"<split> mean TRE: <value>"` (e.g., `test mean TRE: 1.234567`), plus optional per-pair logs if enabled.


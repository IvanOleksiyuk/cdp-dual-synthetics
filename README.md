# CDP Dual Synthetics

Code for reproducing the results of:

> **Authentication of Copy Detection Patterns via Cross-Camera Dual-Synthetic Referencing**
> Ivan Oleksiyuk, Roman Chaban, Slava Voloshynovskiy
> Published in the ICIP 2026 proceedings. [IEEE Xplore](https://ieeexplore.ieee.org/document/11630373)

If you use this code, please cite:
```bibtex
@INPROCEEDINGS{11630373,
  author={Oleksiyuk, Ivan and Chaban, Roman and Voloshynovskiy, Slava},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)},
  title={Authentication of Copy Detection Patterns Via Cross-Camera Dual-Synthetic Referencing},
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Cameras;Printing;Authentication;Signal detection;Printers;Modeling;Measurement;Conferences;Counterfeiting;Probes;Copy Detection Patterns;authentication;synthetic referencing;mutual information;copy attacks},
  doi={10.1109/ICIP61757.2026.11630373}}
```

## Overview

This repository implements a Snakemake pipeline for training and evaluating
cross-camera dual-synthetic reference models for Copy Detection Pattern (CDP)
authentication. It covers data preparation, model training, reference
generation/export, and the evaluation plots/tables used in the paper.

## 1. Prerequisites

- conda (or miniconda)
- A CUDA-capable GPU is strongly recommended for training; the pipeline also
  runs on CPU (much slower) or a weak GPU (see `weak_gpu` in the configs).

## 2. Download the dataset

Download and unzip the dataset from
https://github.com/romaroman/cdp-synthetics-dataset

After unzipping, you should have a file at
`${DATA_DIR}wifs2024dataset/wifs2024dataset/orig_template/rcod/000001.tiff`.

## 3. Set up the environment

```bash
conda env create -f cdp-dual-conda.yml -n cdp-dual
conda activate cdp-dual
```

Define these variables in your shell for the commands below (adjust the
paths for your machine):

```bash
DATA_DIR="/path/to/data/"                 # where you unzipped the dataset
WORKSPACE_DIR="/path/to/workspace/"       # where run outputs will be written
REPO_DIR="/path/to/cdp-dual-synthetics/"  # this repository's path
PREPARED_DATASET_DIR="${DATA_DIR}cdp_transport_dataset_default/"
WANDB_KEY="skip"                          # or your Weights & Biases API key to enable W&B logging
```

`WANDB_KEY="skip"` runs training without Weights & Biases: metrics are
logged to a local `metrics.csv` per model instead, and everything else in
the pipeline (including `collect_important_metrics`) works the same way.

## 4. Prepare the data

Restructures the raw dataset into train/valid/test splits and block-splits
each image (required by the model's data loader). Run once with
`--dry-run` first to check the plan, then again without it:

```bash
python -m snakemake -s workflow/prepare_data.smk --cores="all" \
 --configfile workflow/config/2_device_run.yaml \
 -j 1 \
 --resources gpu=1 \
 --rerun-incomplete -p \
 --config \
 srcdir=${REPO_DIR} \
 wrkspdir=${WORKSPACE_DIR}2_device_run_1/ \
 datadir=${DATA_DIR} \
 prepared_dataset_dir=${PREPARED_DATASET_DIR} \
 --dry-run
# drop --dry-run once the plan looks right
```

This step only needs to be run once; both experiment configs below reuse
the same prepared dataset.

## 5. Quick smoke test (recommended before the full run)

The full run below takes several hours. To first verify your setup works
end-to-end with a tiny config (small data ranges, minimal models, `debug`
mode), run:

```bash
python -m snakemake -s workflow/main.smk --cores="all" \
 --configfile workflow/config/small_integration_test.yaml \
 -j 1 \
 --resources gpu=1 \
 --rerun-incomplete -p \
 --config \
 srcdir=${REPO_DIR} \
 wrkspdir=${WORKSPACE_DIR}test_run/ \
 datadir=${DATA_DIR} \
 wandb_key=${WANDB_KEY} \
 --dry-run
# drop --dry-run once the plan looks right
```

This runs in a few minutes and exercises the whole pipeline (training,
export, evaluation, plots) on a handful of images.

## 6. Reproduce the main-text results (2 devices)

```bash
python -m snakemake -s workflow/main.smk --cores="all" \
 --configfile workflow/config/2_device_run.yaml \
 -j 1 \
 --resources gpu=1 \
 --rerun-incomplete -p \
 --config \
 srcdir=${REPO_DIR} \
 wrkspdir=${WORKSPACE_DIR}2_device_run_1/ \
 datadir=${DATA_DIR} \
 prepared_dataset_dir=${PREPARED_DATASET_DIR} \
 wandb_key=${WANDB_KEY} \
 --dry-run
# drop --dry-run once the plan looks right
```

This trains both models and evaluates them across all 10
`blocks_combine-N` settings in a single run (already the default in
`2_device_run.yaml`'s `reference_eval_mode`). Expect this to take several
hours depending on your GPU. Results (plots and LaTeX tables) are written
under `${WORKSPACE_DIR}2_device_run_1/main_results/` and
`main_results_truncated/`.

`workflow/config/2_device_run_with_roman.yaml` additionally includes a
comparison against a third-party literature baseline ("roman"), whose code
is not part of this repository. To use it, drop your own
`src/configs/2_device_run/roman.yaml` (and matching model implementation)
in place and run with that configfile instead.

## 7. Reproduce the supplementary results (all devices)

Same as step 6, but with all 7 imaging devices instead of 2, and a much
larger job count as a result:

```bash
python -m snakemake -s workflow/main.smk --cores="all" \
 --configfile workflow/config/all_device_run.yaml \
 -j 1 \
 --resources gpu=1 \
 --rerun-incomplete -p \
 --config \
 srcdir=${REPO_DIR} \
 wrkspdir=${WORKSPACE_DIR}all_device_run_1/ \
 datadir=${DATA_DIR} \
 prepared_dataset_dir=${PREPARED_DATASET_DIR} \
 wandb_key=${WANDB_KEY} \
 --dry-run
# drop --dry-run once the plan looks right
```

## Notes

- All commands above are safe to interrupt and resume: rerun the same
  command (`--rerun-incomplete` is already included) and Snakemake will
  only redo what's missing or incomplete. If a run was killed forcefully
  (not a clean interrupt) and you see a `LockException`, run the same
  command with `--unlock` once, then rerun normally.
- Training determinism can be toggled per run via
  `--config deterministic=False` (default is `True`) if you want faster,
  non-reproducible training.
- The `slurm_partition`/`slurm_extra` resource hints in
  `workflow/main.smk` are only used if you run Snakemake with a cluster
  executor plugin; adjust them to your own cluster, or ignore them for
  local execution as shown above.

## License

MIT License, see [LICENSE](LICENSE).

`src/utils/hydra_utils.py` is adapted from
[mattcleigh/mltools](https://github.com/mattcleigh/mltools) (MIT License,
Copyright (c) 2023 Matthew Leigh).

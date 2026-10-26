#!/usr/bin/env bash
# scripts/install.sh
# ====================
# Installs the production runtime dependencies for this repo. Used by
# render.yaml's buildCommand and recommended for local setup too (see
# root README.md and web_prototype/api/README.md).
#
# Why this isn't just `pip install -r requirements.txt`:
#
# xgboost's published Linux wheel unconditionally declares a dependency on
# nvidia-nccl-cu13 (verified via `pip show xgboost` -> Requires-Dist:
# nvidia-nccl-cu13; platform_system == "Linux"). That package exists for
# multi-GPU distributed training and is ~290MB. This repo only ever does
# CPU inference and CPU cross-validated training (see inference.py's
# module docstring and blue_team_pipeline.py) -- nothing here uses a GPU
# or NCCL. Pulling it in anyway roughly quadruples the installed size
# (verified: ~310MB with this script's approach vs. ~1.2GB with a plain
# `pip install -r requirements.txt xgboost==3.4.1`), which matters
# directly for Render free-tier build time and disk.
#
# `pip install --no-deps` skips xgboost's declared dependencies entirely.
# That's safe here specifically because xgboost's only OTHER real runtime
# dependencies are numpy and scipy (see the same `pip show` output), both
# of which are already installed by the first step below. Verified working
# end-to-end: DMatrix construction + xgb.train() with this exact install
# order and no nvidia packages present.
#
# If you ever add code that needs xgboost's GPU/distributed features,
# remove the --no-deps line below and accept the larger install.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

pip install -r requirements.txt
pip install --no-deps xgboost==3.4.1

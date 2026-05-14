#!/usr/bin/env bash
#SBATCH --job-name=sls-rolora
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=slurm-%j.out

# DelftBlue MIG slice (10 GB VRAM, ≤4h hard cap). Default partition for RoBERTa-Large LoRA
# work — the deep-research plan estimates the model fits in ≤6 GB at fp16/bs=16.
#
# NOTE: partition / module names are UNCONFIRMED until TAs (Heijmans / Wang) share the
# DelftBlue access instructions. See docs/setup/delftblue.md.
#
# Usage:
#   sbatch slurm/gpu-a100-small.sbatch experiments/configs/<file>.yaml

set -euo pipefail

CONFIG_PATH="${1:?usage: sbatch slurm/gpu-a100-small.sbatch <config.yaml>}"
cd "$SLURM_SUBMIT_DIR"

# Modules — TBD after TA instructions. Placeholders below.
# module load 2024r1
# module load python/3.11

# Cache datasets on /scratch to avoid re-tokenizing per job.
export SLS_DATA_DIR="${SLS_DATA_DIR:-/scratch/$USER/sls-data}"
export SLS_RESULTS_DIR="${SLS_RESULTS_DIR:-/scratch/$USER/sls-results}"

echo "job $SLURM_JOB_ID on $(hostname) — config=$CONFIG_PATH"
echo "data dir: $SLS_DATA_DIR"

# Entry point lands in the next iteration; for now this validates queue + env.
uv run python -m sls_rolora.cli --config "$CONFIG_PATH" || {
    rc=$?
    echo "training entrypoint not yet implemented (expected during pre-launch)."
    exit $rc
}

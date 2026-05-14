#!/usr/bin/env bash
#SBATCH --job-name=sls-rolora
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# DelftBlue V100 (32 GB) — backup partition when gpu-a100-small queue is saturated.
# Longer time limit, more VRAM headroom for ranks 8+ or longer sequences.
#
# NOTE: partition / module names UNCONFIRMED — see docs/setup/delftblue.md.
#
# Usage:
#   sbatch slurm/gpu-v100.sbatch experiments/configs/<file>.yaml

set -euo pipefail

CONFIG_PATH="${1:?usage: sbatch slurm/gpu-v100.sbatch <config.yaml>}"
cd "$SLURM_SUBMIT_DIR"

# module load 2024r1
# module load python/3.11

export SLS_DATA_DIR="${SLS_DATA_DIR:-/scratch/$USER/sls-data}"
export SLS_RESULTS_DIR="${SLS_RESULTS_DIR:-/scratch/$USER/sls-results}"

echo "job $SLURM_JOB_ID on $(hostname) — config=$CONFIG_PATH"

uv run python -m sls_rolora.cli --config "$CONFIG_PATH" || {
    rc=$?
    echo "training entrypoint not yet implemented (expected during pre-launch)."
    exit $rc
}

#!/bin/bash
#
# QNLI reproduction — cell C2 (20 clients, RoBERTa-Large, LoRA rank 4) — FFA-LoRA mode.
# A is frozen at init; only B is trained and aggregated.
# Config: experiments/configs/repro_qnli_c20_r4.yaml
#
# Submit (seed 0 default):
#   sbatch slurm/repro_qnli_c20_r4_ffa_lora.sbatch
# Submit with a non-default seed:
#   SEED=1 sbatch slurm/repro_qnli_c20_r4_ffa_lora.sbatch
#
# See slurm/repro_qnli_c20_r4_rolora.sbatch for sizing rationale and prereqs.
#
#SBATCH --job-name=sls-c20-r4-ffa_lora
#SBATCH --partition=gpu-a100-small
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=03:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2      # gpu-a100-small caps cpus-per-task at 2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M    # gpu-a100-small caps mem-per-cpu at 8000 MB (8G is rejected)
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Always operate from the directory where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p slurm_logs results

# --- Modules ---------------------------------------------------------------
module load 2024r1
module load cuda

# --- Environment -----------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
export WANDB_CONSOLE=off

# HuggingFace cache on scratch — must be pre-warmed on the login node via
# `bash scripts/warm_caches.sh`. DelftBlue compute nodes have NO outbound
# internet; offline flags prevent revision-check HEAD requests that would fail.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

set -a; [ -f .env ] && source .env; set +a
: "${WANDB_API_KEY:?FAIL: WANDB_API_KEY missing — populate .env on the cluster}"

# --- Pinned per-file -------------------------------------------------------
export SLS_ALTERNATION_MODE=ffa_lora
CONFIG_PATH="experiments/configs/repro_qnli_c20_r4.yaml"
SEED="${SEED:-0}"

# --- Wandb identity --------------------------------------------------------
export WANDB_ENTITY="scalable-learning-7"
export WANDB_PROJECT="sls-rolora-repro"
export WANDB_RUN_GROUP="qnli_c20_r4"
export WANDB_NAME="${SLS_ALTERNATION_MODE}_seed${SEED}"
export WANDB_TAGS="qnli,c20,r4,${SLS_ALTERNATION_MODE},seed${SEED}"

# --- Sanity checks ---------------------------------------------------------
echo "Node:           $(hostname)"
echo "Job ID:         ${SLURM_JOB_ID:-N/A}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Allocated GPUs: ${SLURM_GPUS_PER_TASK:-?}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "Working dir:    $(pwd)"
echo "Mode:           $SLS_ALTERNATION_MODE"
echo "Config:         $CONFIG_PATH"
echo "Seed:           $SEED"
echo "Wandb run:      $WANDB_PROJECT/$WANDB_RUN_GROUP/$WANDB_NAME"
nvidia-smi || true

command -v uv >/dev/null || { echo "FAIL: uv not on PATH"; exit 1; }
[ -f "$CONFIG_PATH" ] || { echo "FAIL: config missing at $CONFIG_PATH"; exit 1; }
[ -x code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python ] \
    || { echo "FAIL: supplement venv missing — run scripts/install_supplement.sh on login node"; exit 1; }
code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python -c \
    "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "FAIL: torch.cuda.is_available() is False — check cuda module / torch build"; exit 1; }
code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python -c \
    "import wandb" \
    || { echo "FAIL: wandb missing in supplement venv — re-run scripts/install_supplement.sh"; exit 1; }
echo "Sanity checks passed."

# --- Run -------------------------------------------------------------------
CONFIG="$CONFIG_PATH" SEED="$SEED" \
    srun bash scripts/smoke_supplement.sh "$SLS_ALTERNATION_MODE"

#!/bin/bash
#
# QNLI lr sweep — 50 clients, RoBERTa-base, LoRA rank 4, 20 rounds — RoLoRA mode.
# Config: experiments/configs/repro_qnli_c50_r4_lr*.yaml
#
# Submit (seed 0 default):
#   sbatch slurm/repro_qnli_c20_r4_rolora.sbatch
# Submit with a non-default seed:
#   SEED=1 sbatch slurm/repro_qnli_c20_r4_rolora.sbatch
#
# Sizing rationale (gpu-a100-small, 10 GB MIG slice, ≤4h cap):
#   - Single Python process. No torch.multiprocessing, no DataLoader workers
#     (num_workers=0), share_local_model=True serialises clients, OMP/MKL
#     pinned to 1. cpus-per-task=2 covers the main process plus a sliver of
#     scheduling slack — gpu-a100-small caps cpus-per-task at 2.
#   - 2 cpus × 8000M ≈ 16 GB total (partition caps both knobs). Budget:
#     RoBERTa-Large fp32 ~1.4 GB + activations ~2 GB + HF caches ~3 GB +
#     PyTorch overhead ~2 GB + FederatedScope state — tight but adequate.
#     If we OOM on host RAM, fallback is gpu-a100 (full A100, 24h cap) where
#     mem-per-cpu and cpus-per-task aren't as constrained.
#   - --time=03:59:59 maximises the 4h MIG slice; estimated job duration is
#     ~30 min (30 rounds × 20 local steps × 20 clients × bs=32).
#
# Prerequisites (one-time on the DelftBlue login node) — see docs/setup/delftblue.md:
#   1) mkdir -p slurm_logs
#   2) `.env` populated with WANDB_API_KEY
#   3) uv installed and on PATH
#   4) uv sync   (creates project .venv)
#   5) bash scripts/install_supplement.sh   (creates supplement .venv-supplement)
#   6) sbatch slurm/warm_caches.sbatch   (downloads model + qnli.json to scratch)
#
#SBATCH --job-name=sls-c50-r4-rolora-lr2e-2
#SBATCH --partition=gpu-a100-small
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=03:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2      # gpu-a100-small caps cpus-per-task at 2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M    # 2 cpus × 8000M ≈ 16G total. gpu-a100-small caps both cpus
                               # at 2 and mem-per-cpu at 8000 MB. Budget: model+activations+HF+overhead ~12G; tight.
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Always operate from the directory where sbatch was invoked. Slurm sets
# SLURM_SUBMIT_DIR; together with the prereq that you submit from the repo
# root, this keeps all the relative paths below correct.
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
# internet ([Errno 101] Network is unreachable for huggingface.co); we set
# HF offline flags below so HF doesn't try sneaky revision-check HEAD
# requests and instead just reads what's in HF_HOME.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Source .env for WANDB_API_KEY (and fail-fast if absent).
set -a; [ -f .env ] && source .env; set +a
: "${WANDB_API_KEY:?FAIL: WANDB_API_KEY missing — populate .env on the cluster}"

# --- Pinned per-file -------------------------------------------------------
export SLS_ALTERNATION_MODE=rolora
CONFIG_PATH="experiments/configs/repro_qnli_c50_r4_lr2e-2.yaml"
SEED="${SEED:-0}"

# --- Wandb identity --------------------------------------------------------
export WANDB_ENTITY="scalable-learning-7"
export WANDB_PROJECT="sls-rolora-repro"
export WANDB_RUN_GROUP="qnli_c50_r4"
export WANDB_NAME="${SLS_ALTERNATION_MODE}_lr2e-2_seed${SEED}"
export WANDB_TAGS="qnli,c50,r4,${SLS_ALTERNATION_MODE},lr2e-2,seed${SEED}"

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
# Invoke our canonical job body. It handles log naming, manifest header,
# pass/fail markers, and the SEED override.
CONFIG="$CONFIG_PATH" SEED="$SEED" \
    srun bash scripts/smoke_supplement.sh "$SLS_ALTERNATION_MODE"

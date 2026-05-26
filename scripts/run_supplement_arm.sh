#!/usr/bin/env bash
# Cross-platform single-arm supplement runner (macOS / Linux / Windows git-bash).
#
# The canonical scripts/smoke_supplement.sh is fine for one-off smokes, but for
# the overnight mechanism matrix we want: a per-arm TAG for the log filename (so
# parallel arms don't clash), arbitrary FederatedScope cfg overrides per arm,
# and wandb wired on by default. This runner adds those and resolves the venv
# python on either layout ($VENV/bin/python on POSIX, $VENV/Scripts/python.exe
# on Windows).
#
# Usage:
#   MODE=rolora TAG=rolora_sgd bash scripts/run_supplement_arm.sh \
#       experiments/configs/overnight_local_qnli.yaml
#   MODE=rolora TAG=rolora_adamw bash scripts/run_supplement_arm.sh \
#       experiments/configs/overnight_local_qnli.yaml \
#       train.optimizer.type AdamW train.optimizer.lr 0.0005
#
# Device note: the overnight config requests use_gpu/device 0. On a CUDA box
# that's the GPU; on Apple silicon the supplement falls back to MPS then CPU.
# To force a device, pass e.g. `use_gpu False device -1` as trailing overrides.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SUPP="$REPO/code/harness/rolora-supplement/RoLoRA-code"
VENV="$SUPP/.venv-supplement"
PY="$VENV/bin/python"
[ -x "$PY" ] || PY="$VENV/Scripts/python.exe"
RESULTS="$REPO/results"

CONFIG="${1:?usage: MODE=.. TAG=.. bash scripts/run_supplement_arm.sh <config.yaml> [cfg overrides...]}"
shift || true
OVERRIDES=("$@")

MODE="${MODE:-rolora}"
case "$MODE" in rolora|lora|ffa_lora) ;; *) echo "bad MODE=$MODE"; exit 2;; esac
TAG="${TAG:-$MODE}"

[ -x "$PY" ] || { echo "error: supplement venv python missing ($VENV/{bin,Scripts}) — run scripts/install_supplement.sh"; exit 1; }
[ -f "$REPO/$CONFIG" ] || [ -f "$CONFIG" ] || { echo "error: config not found: $CONFIG"; exit 1; }

mkdir -p "$RESULTS"
log="$RESULTS/overnight_${TAG}.log"
echo "[run] mode=$MODE tag=$TAG py=$PY -> $log"
echo "      overrides: ${OVERRIDES[*]:-(none)}"

{
  echo "# repo: $REPO"
  echo "# git_sha: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# config: $CONFIG"
  echo "# mode: $MODE"
  echo "# tag: $TAG"
  echo "# overrides: ${OVERRIDES[*]:-(none)}"
  echo "# started_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cd "$REPO"
  # wandb: default to OFFLINE for unattended runs (no interactive auth prompt /
  # no creds required). Offline still creates wandb.run, so the mechanism and
  # train probes fire and buffer to $WANDB_DIR/wandb/offline-run-*; run
  # `wandb sync` later to upload. Set WANDB_MODE=online (with creds) to stream.
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_DIR="${WANDB_DIR:-$RESULTS}"
  export WANDB_ENTITY="${WANDB_ENTITY:-scalable-learning-7}"
  export WANDB_PROJECT="${WANDB_PROJECT:-sls-rolora-local}"
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-overnight_local_qnli}"
  export WANDB_NAME="${WANDB_NAME:-$TAG}"
  export WANDB_TAGS="${WANDB_TAGS:-local,qnli,roberta-base,$MODE}"
  NO_COLOR=1 SLS_ALTERNATION_MODE="$MODE" \
    "$PY" scripts/run_supplement.py --cfg "$CONFIG" "${OVERRIDES[@]}"
} >"$log" 2>&1 && echo "[done] $TAG (exit 0)" || { echo "[FAIL] $TAG; tail:"; tail -40 "$log"; exit 1; }

#!/usr/bin/env bash
set -u
cd /Users/vliftode/personal/scalable-learning || exit 1
WATCH_PID="${1:-}"
{
  echo "[queue] watcher pid=$$ waiting for active orthogonal-A seed0 PID=${WATCH_PID:-none}: $(date)"
  if [[ -n "$WATCH_PID" ]]; then
    while kill -0 "$WATCH_PID" 2>/dev/null; do
      sleep 45
    done
  fi
  echo "[queue] starting BBA phase-controller run: $(date)"
  WANDB_MODE=online \
  WANDB_PROJECT=sls-rolora-repro \
  WANDB_RUN_GROUP=qnli_c50_r4_improvements \
  WANDB_TAGS=local,proxy,phase-controller,bba,orthogonal_a,qnli,roberta-base,rolora \
  SLS_LORA_INIT=orthogonal_a \
  SLS_PHASE_PATTERN=BBA \
  SLS_DEVICE=mps \
  SLS_MONITOR=1 \
  MODE=rolora \
  TAG=proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0 \
  SEED=0 \
    bash scripts/run_supplement_arm.sh experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml seed 0
  status=$?
  echo "[queue] completed BBA phase-controller run status=$status: $(date)"
  exit "$status"
} >> results/queue_proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed0.log 2>&1

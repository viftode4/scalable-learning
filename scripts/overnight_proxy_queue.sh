#!/usr/bin/env bash
# Sequential overnight queue for local MPS proxy experiments.
# It intentionally runs one training job at a time and uses W&B only for real runs.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

CONFIG="experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml"
RESULTS="$REPO/results"
mkdir -p "$RESULTS"

QUEUE_LOG="$RESULTS/overnight_proxy_queue_$(date -u +%Y%m%dT%H%M%SZ).log"
LATEST_LOG="$RESULTS/overnight_proxy_queue_latest.log"
exec > >(tee -a "$QUEUE_LOG" "$LATEST_LOG") 2>&1

RUN_GROUP="${WANDB_RUN_GROUP:-qnli_c50_r4_improvements}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-sls-rolora-repro}"
WANDB_ENTITY_NAME="${WANDB_ENTITY:-scalable-learning-7}"
WAIT_PIDS_RAW="${SLS_WAIT_PIDS:-}"

log() { printf '# %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

wait_for_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if ps -p "$pid" >/dev/null 2>&1; then
    log "waiting_for_existing_process pid=$pid"
    while ps -p "$pid" >/dev/null 2>&1; do
      sleep 60
    done
    log "existing_process_finished pid=$pid"
  else
    log "existing_process_not_running pid=$pid"
  fi
}

already_done() {
  local tag="$1"
  local log_file="$RESULTS/overnight_${tag}.log"
  local evidence_csv="$REPO/evidence/improvement_diagnostics_20260604/${tag}/server_metrics.csv"
  if [ -f "$evidence_csv" ] && python3 - "$evidence_csv" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))

if len(rows) >= 20 and rows[-1].get("round") == "19":
    raise SystemExit(0)
raise SystemExit(1)
PY
  then
    return 0
  fi
  [[ -f "$log_file" ]] && grep -q "\[done\] ${tag} (exit 0)" "$log_file"
}

run_arm() {
  local tag="$1"
  local seed="$2"
  local init="$3"
  local phase_policy="$4"
  local phase_pattern="$5"
  local tags="$6"

  if already_done "$tag"; then
    log "skip_done tag=$tag"
    return 0
  fi

  local state_file="$RESULTS/sls_phase_state_${tag}.json"
  rm -f "$state_file"

  log "start tag=$tag seed=$seed init=$init policy=$phase_policy pattern=${phase_pattern:-none}"
  env \
    WANDB_MODE=online \
    WANDB_PROJECT="$WANDB_PROJECT_NAME" \
    WANDB_ENTITY="$WANDB_ENTITY_NAME" \
    WANDB_RUN_GROUP="$RUN_GROUP" \
    WANDB_TAGS="$tags" \
    SLS_LORA_INIT="$init" \
    SLS_PHASE_POLICY="$phase_policy" \
    SLS_PHASE_PATTERN="$phase_pattern" \
    SLS_ADAPTIVE_MIN_B_ROUNDS="${SLS_ADAPTIVE_MIN_B_ROUNDS:-2}" \
    SLS_ADAPTIVE_MAX_B_ROUNDS="${SLS_ADAPTIVE_MAX_B_ROUNDS:-4}" \
    SLS_ADAPTIVE_VAL_GAIN_EPSILON="${SLS_ADAPTIVE_VAL_GAIN_EPSILON:-0.001}" \
    SLS_PHASE_STATE_FILE="$state_file" \
    SLS_DEVICE="${SLS_DEVICE:-mps}" \
    SLS_MONITOR="${SLS_MONITOR:-1}" \
    MODE=rolora \
    TAG="$tag" \
    SEED="$seed" \
    bash scripts/run_supplement_arm.sh "$CONFIG" seed "$seed"
  log "finish tag=$tag"
}

log "queue_start repo=$REPO git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "queue_log=$QUEUE_LOG"
log "wandb_entity=$WANDB_ENTITY_NAME wandb_project=$WANDB_PROJECT_NAME group=$RUN_GROUP"
log "tests_smokes_are_wandb_disabled_in scripts/smoke_supplement.sh"

for pid in $WAIT_PIDS_RAW; do
  wait_for_pid "$pid"
done

# Highest-value overnight set:
# 1) adaptive-refresh seeds: tests the novel dynamic-basis idea against fixed BBA.
# 2) BBA seed2: completes 3-seed fixed-schedule evidence with existing seed0 + active seed1.
# 3) orth-A AB seeds: isolates initialization effect if there is time left.
run_arm "proxy_adaptive_refresh_orth_a_c50_r4_lr1e-2_seed0" 0 "orthogonal_a" "adaptive_refresh" "" "local,proxy,adaptive-refresh,orthogonal_a,qnli,roberta-base,rolora,monitor"
run_arm "proxy_adaptive_refresh_orth_a_c50_r4_lr1e-2_seed1" 1 "orthogonal_a" "adaptive_refresh" "" "local,proxy,adaptive-refresh,orthogonal_a,qnli,roberta-base,rolora,monitor,replication"
run_arm "proxy_adaptive_refresh_orth_a_c50_r4_lr1e-2_seed2" 2 "orthogonal_a" "adaptive_refresh" "" "local,proxy,adaptive-refresh,orthogonal_a,qnli,roberta-base,rolora,monitor,replication"
run_arm "proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed2" 2 "orthogonal_a" "default" "BBA" "local,proxy,phase-controller,bba,orthogonal_a,qnli,roberta-base,rolora,monitor,replication"
run_arm "proxy_orth_a_c50_r4_lr1e-2_seed1" 1 "orthogonal_a" "default" "" "local,proxy,orthogonal_a,qnli,roberta-base,rolora,monitor,ab,replication"
run_arm "proxy_orth_a_c50_r4_lr1e-2_seed2" 2 "orthogonal_a" "default" "" "local,proxy,orthogonal_a,qnli,roberta-base,rolora,monitor,ab,replication"

log "queue_complete"

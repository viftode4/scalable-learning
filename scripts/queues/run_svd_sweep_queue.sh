#!/usr/bin/env bash
set -u

CONFIG="experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml"
COMMON_TAGS="local,proxy,qnli,roberta-base,c50,r4,rolora,svd_compensated,monitor"

run_arm() {
  local tag="$1" seed="$2" phase_pattern="$3" tags="$4"
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') START $tag seed=$seed phase=${phase_pattern:-default} ====="
  unset SLS_PHASE_POLICY
  unset SLS_PHASE_PATTERN
  export SLS_LORA_INIT=svd_compensated
  export SLS_MONITOR=1
  export SLS_DEVICE=mps
  export MODE=rolora
  export TAG="$tag"
  export SEED="$seed"
  export WANDB_MODE=online
  export WANDB_ENTITY=scalable-learning-7
  export WANDB_PROJECT=sls-rolora-repro
  export WANDB_RUN_GROUP=qnli_c50_r4_svd_sweep
  export WANDB_NAME="$tag"
  export WANDB_TAGS="$tags"
  if [[ -n "$phase_pattern" ]]; then
    export SLS_PHASE_PATTERN="$phase_pattern"
  fi
  bash scripts/run_supplement_arm.sh "$CONFIG" seed "$seed"
  local rc=$?
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') END $tag rc=$rc ====="
  return "$rc"
}

failures=0

# Plain SVD compensated: seed0 already completed cleanly as k8tr2lmw, run the missing CI seeds.
run_arm "proxy_svd_compensated_c50_r4_lr1e-2_seed1" 1 "" "$COMMON_TAGS,seed1" || failures=$((failures+1))
run_arm "proxy_svd_compensated_c50_r4_lr1e-2_seed2" 2 "" "$COMMON_TAGS,seed2" || failures=$((failures+1))

# Daniel-style follow-up: combine the compensated SVD basis with the stronger BBA phase controller.
run_arm "proxy_svd_compensated_bba_c50_r4_lr1e-2_seed0" 0 "BBA" "$COMMON_TAGS,bba,phase-controller,seed0" || failures=$((failures+1))
run_arm "proxy_svd_compensated_bba_c50_r4_lr1e-2_seed1" 1 "BBA" "$COMMON_TAGS,bba,phase-controller,seed1" || failures=$((failures+1))
run_arm "proxy_svd_compensated_bba_c50_r4_lr1e-2_seed2" 2 "BBA" "$COMMON_TAGS,bba,phase-controller,seed2" || failures=$((failures+1))

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') SVD SWEEP DONE failures=$failures ====="
exit "$failures"

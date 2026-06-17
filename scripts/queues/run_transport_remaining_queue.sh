#!/usr/bin/env bash
set -u

# Remaining basis-transport runs after the 2026-06-12 battery/session kill:
# P1 (proxy_transport_c50_r4_lr1e-2_seed0) completed at 0.8779 and is NOT rerun.
# P2 died at round 7/20 and restarts from scratch; P3 never started.
# See .plans/lora-basis-transport.md.

CONFIG="experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml"
COMMON_TAGS="local,proxy,qnli,roberta-base,c50,r4,rolora,transport,monitor"

run_arm() {
  local tag="$1" seed="$2" phase_pattern="$3" lora_init="$4" tags="$5"
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') START $tag seed=$seed phase=${phase_pattern:-default} init=${lora_init:-default} ====="
  unset SLS_PHASE_POLICY
  unset SLS_PHASE_PATTERN
  unset SLS_LORA_INIT
  export SLS_LORA_TRANSPORT=ls
  export SLS_MONITOR=1
  export SLS_DEVICE=mps
  export MODE=rolora
  export TAG="$tag"
  export SEED="$seed"
  export WANDB_MODE=online
  export WANDB_ENTITY=scalable-learning-7
  export WANDB_PROJECT=sls-rolora-repro
  export WANDB_RUN_GROUP=qnli_c50_r4_transport
  export WANDB_NAME="$tag"
  export WANDB_TAGS="$tags"
  if [[ -n "$phase_pattern" ]]; then
    export SLS_PHASE_PATTERN="$phase_pattern"
  fi
  if [[ -n "$lora_init" ]]; then
    export SLS_LORA_INIT="$lora_init"
  fi
  bash scripts/run_supplement_arm.sh "$CONFIG" seed "$seed"
  local rc=$?
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') END $tag rc=$rc ====="
  return "$rc"
}

failures=0

# P2 restart: BBA + orth-A + transport vs BBA + orth-A controls (0.885 / 0.881).
run_arm "proxy_transport_bba_orth_a_c50_r4_lr1e-2_seed0" 0 "BBA" "orthogonal_a" "$COMMON_TAGS,bba,orthogonal_a,seed0,restart" || failures=$((failures+1))

# P3: svd_compensated at lr=1e-2 + transport vs its unstable controls.
run_arm "proxy_transport_svd_compensated_c50_r4_lr1e-2_seed0" 0 "" "svd_compensated" "$COMMON_TAGS,svd_compensated,seed0" || failures=$((failures+1))

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') TRANSPORT REMAINING QUEUE DONE failures=$failures ====="
exit "$failures"

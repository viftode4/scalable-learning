#!/usr/bin/env bash
set -u
CONFIG="experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml"
COMMON_TAGS="local,proxy,qnli,roberta-base,c50,r4,rolora,svd_compensated,monitor,bba,phase-controller,retry"
run_arm() {
  local tag="$1" seed="$2" tags="$3"
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') START $tag seed=$seed phase=BBA ====="
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
  export WANDB_RUN_GROUP=qnli_c50_r4_svd_bba_retry
  export WANDB_NAME="$tag"
  export WANDB_TAGS="$tags"
  export SLS_PHASE_PATTERN="BBA"
  bash scripts/run_supplement_arm.sh "$CONFIG" seed "$seed"
  local rc=$?
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') END $tag rc=$rc ====="
  return "$rc"
}
failures=0
run_arm "proxy_svd_compensated_bba_c50_r4_lr1e-2_seed0" 0 "$COMMON_TAGS,seed0" || failures=$((failures+1))
run_arm "proxy_svd_compensated_bba_c50_r4_lr1e-2_seed1" 1 "$COMMON_TAGS,seed1" || failures=$((failures+1))
run_arm "proxy_svd_compensated_bba_c50_r4_lr1e-2_seed2" 2 "$COMMON_TAGS,seed2" || failures=$((failures+1))
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') SVD BBA RETRY DONE failures=$failures ====="
exit "$failures"

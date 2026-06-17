#!/usr/bin/env bash
set -u

# Partial-participation DE-RISK probe (see .plans/partial-participation-rolora.md).
# This is the gate: it decides whether the full p x alpha grid is worth running.
#
# Cells (QNLI / RoBERTa-base / C50 / r4 / lr1e-2 / lda alpha=0.5 / seed 0):
#   1. A1 faithful RoLoRA, FULL participation (p=1.0)  -> degradation reference
#   2. A1 faithful RoLoRA, PARTIAL (p=0.3)             -> does the method drop?
#   3. A0 naive RoLoRA,    PARTIAL (p=0.3)             -> P3: phase-desync failure
#
# Pass conditions to continue to the grid:
#   (i)  monitor shows all sampled clients agree on phase each round under A1
#        (faithfulness fix works);
#   (ii) cell 2 (A1 p=0.3) is materially below cell 1 (A1 p=1.0).
# If A1 does NOT degrade, KILL the campaign and ledger the robustness negative.
# A0 < A1 at p=0.3 confirms the desync artifact (reported, never credited to us).

CONFIG="experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2_pp.yaml"
COMMON_TAGS="local,proxy,qnli,roberta-base,c50,r4,rolora,partial_participation,lda_a0.5,monitor"

run_arm() {
  local tag="$1" seed="$2" rate="$3" round_source="$4" tags="$5"
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') START $tag seed=$seed rate=$rate source=${round_source:-step_count} ====="
  unset SLS_PHASE_POLICY SLS_PHASE_PATTERN SLS_LORA_INIT SLS_LORA_TRANSPORT
  unset SLS_PHASE_ROUND_SOURCE
  if [[ -n "$round_source" ]]; then
    export SLS_PHASE_ROUND_SOURCE="$round_source"
  fi
  export SLS_MONITOR=1
  export SLS_DEVICE=mps
  export MODE=rolora
  export TAG="$tag"
  export SEED="$seed"
  export WANDB_MODE=online
  export WANDB_ENTITY=scalable-learning-7
  export WANDB_PROJECT=sls-rolora-repro
  export WANDB_RUN_GROUP=qnli_c50_r4_partial_participation
  export WANDB_NAME="$tag"
  export WANDB_TAGS="$tags"
  bash scripts/run_supplement_arm.sh "$CONFIG" seed "$seed" \
    federate.sample_client_rate "$rate"
  local rc=$?
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') END $tag rc=$rc ====="
  return "$rc"
}

failures=0

# Cell 1: A1 faithful, full participation -> reference point at alpha=0.5.
run_arm "pp_a1_faithful_p1.0_a0.5_seed0" 0 1.0 global "$COMMON_TAGS,a1,p1.0,seed0" || failures=$((failures+1))

# Cell 2: A1 faithful, partial participation -> the decisive degradation test.
run_arm "pp_a1_faithful_p0.3_a0.5_seed0" 0 0.3 global "$COMMON_TAGS,a1,p0.3,seed0" || failures=$((failures+1))

# Cell 3: A0 naive (step_count phase) at p=0.3 -> quantify the desync failure.
run_arm "pp_a0_naive_p0.3_a0.5_seed0" 0 0.3 "" "$COMMON_TAGS,a0,p0.3,seed0" || failures=$((failures+1))

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') PP DE-RISK QUEUE DONE failures=$failures ====="
exit "$failures"

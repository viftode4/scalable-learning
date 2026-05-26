#!/usr/bin/env bash
# Overnight local mechanism matrix: 6 arms = 3 modes × 2 optimizers, run
# 2-in-parallel per wave on the laptop GPU. Produces accuracy-vs-round +
# train-loss + classifier/A/B-norm traces (wandb offline + results/*.log).
#
#   wave 1:  rolora   (authors' SGD)    ‖  lora     (authors' SGD)
#   wave 2:  ffa_lora (authors' SGD)    ‖  rolora   (AdamW lr 5e-4)
#   wave 3:  lora     (AdamW lr 5e-4)   ‖  ffa_lora (AdamW lr 5e-4)
#
# Background: a single-arm probe (rolora_adamw, 40 rounds) reached test_acc
# 0.8766 while the authors' SGD setup sits at chance after the same budget.
# This matrix repeats the SGD baselines on the fixed trainer (no eval-mode
# step_count drift, alternation pre-optimizer) and extends the AdamW probe to
# all three modes so the next cluster submission can pick an optimizer based
# on actual per-mode evidence rather than a single arm.
#
# Run unattended:  bash scripts/overnight_matrix.sh
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
CFG="experiments/configs/overnight_local_qnli.yaml"
ADAMW_OVERRIDES=(train.optimizer.type AdamW train.optimizer.lr 0.0005)

run() {  # run MODE TAG [cfg overrides...]
  MODE="$1" TAG="$2" bash scripts/run_supplement_arm.sh "$CFG" "${@:3}"
}

echo "=== WAVE 1: rolora_sgd || lora_sgd ==="
run rolora rolora_sgd & P1=$!
run lora   lora_sgd   & P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "wave1 exit codes: rolora_sgd=$R1 lora_sgd=$R2"

echo "=== WAVE 2: ffa_lora_sgd || rolora_adamw ==="
run ffa_lora ffa_lora_sgd & P3=$!
run rolora   rolora_adamw "${ADAMW_OVERRIDES[@]}" & P4=$!
wait $P3; R3=$?
wait $P4; R4=$?
echo "wave2 exit codes: ffa_lora_sgd=$R3 rolora_adamw=$R4"

echo "=== WAVE 3: lora_adamw || ffa_lora_adamw ==="
run lora     lora_adamw     "${ADAMW_OVERRIDES[@]}" & P5=$!
run ffa_lora ffa_lora_adamw "${ADAMW_OVERRIDES[@]}" & P6=$!
wait $P5; R5=$?
wait $P6; R6=$?
echo "wave3 exit codes: lora_adamw=$R5 ffa_lora_adamw=$R6"

echo "OVERNIGHT_MATRIX_DONE r1=$R1 r2=$R2 r3=$R3 r4=$R4 r5=$R5 r6=$R6"

#!/usr/bin/env bash
# Overnight local mechanism matrix: 4 arms, 2 in parallel per wave, on the
# laptop GPU. Produces accuracy-vs-round + train-loss + classifier/A/B-norm
# traces (wandb offline + results/*.log) for:
#   wave 1:  rolora (authors' SGD)      ‖  lora (authors' SGD)
#   wave 2:  ffa_lora (authors' SGD)    ‖  rolora (AdamW lr 5e-4)
#
# Arms 1-3 test "does the classifier-unfreeze fix make each mode learn off
# 0.50". Arm 4 tests the undertraining/weak-optimizer hypothesis (SGD 0.005 vs
# AdamW 5e-4) on rolora.
#
# Run unattended:  bash scripts/overnight_matrix.sh   (use run_in_background)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
CFG="experiments/configs/overnight_local_qnli.yaml"

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
run rolora   rolora_adamw train.optimizer.type AdamW train.optimizer.lr 0.0005 & P4=$!
wait $P3; R3=$?
wait $P4; R4=$?
echo "wave2 exit codes: ffa_lora_sgd=$R3 rolora_adamw=$R4"

echo "OVERNIGHT_MATRIX_DONE r1=$R1 r2=$R2 r3=$R3 r4=$R4"

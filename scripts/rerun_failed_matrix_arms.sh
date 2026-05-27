#!/usr/bin/env bash
# Re-run the 4 arms that the first overnight_matrix.sh dispatch lost to two
# infra bugs (now fixed in scripts/run_supplement_arm.sh):
#   - bash `set -u` + `${OVERRIDES[@]}` empty-array unbound-variable (the 3
#     SGD arms all crashed before training started)
#   - FederatedScope auto-generates `exp/.../sub_exp_<UTC-seconds>` so two
#     arms launched in the same second collided on the same directory and
#     the second one died with FileExistsError (lora_adamw vs ffa_lora_adamw)
# Both fixes landed on main as part of the overnight-evidence follow-up.
#
# This script only re-runs the arms that did not produce evidence the first
# time. rolora_adamw (0.8766) and ffa_lora_adamw (already running / finished
# by hand) are preserved from the original matrix and not re-run.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
CFG="experiments/configs/overnight_local_qnli.yaml"
ADAMW_OVERRIDES=(train.optimizer.type AdamW train.optimizer.lr 0.0005)

run() {  # run MODE TAG [cfg overrides...]
  MODE="$1" TAG="$2" bash scripts/run_supplement_arm.sh "$CFG" "${@:3}"
}

echo "=== RERUN WAVE 1: rolora_sgd || lora_sgd ==="
run rolora rolora_sgd & P1=$!
run lora   lora_sgd   & P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "rerun-wave1 exit codes: rolora_sgd=$R1 lora_sgd=$R2"

echo "=== RERUN WAVE 2: ffa_lora_sgd || lora_adamw ==="
run ffa_lora ffa_lora_sgd & P3=$!
run lora     lora_adamw "${ADAMW_OVERRIDES[@]}" & P4=$!
wait $P3; R3=$?
wait $P4; R4=$?
echo "rerun-wave2 exit codes: ffa_lora_sgd=$R3 lora_adamw=$R4"

echo "RERUN_MATRIX_DONE r1=$R1 r2=$R2 r3=$R3 r4=$R4"

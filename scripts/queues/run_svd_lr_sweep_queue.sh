#!/usr/bin/env bash
# SVD-compensated init — learning-rate verification sweep (reviewer request).
#
# WHY THIS EXISTS
# ---------------
# Reviewer feedback (2026-06-17): "SVD initialization improves 0.6% on QNLI ...
# you need to do more learning-rate tuning experiments ... to verify this
# limited improvement." The +0.6 pp is the 3-seed proxy result at lr=1e-2 ONLY
# (SVD 85.70 ± 0.59 vs vanilla RoLoRA 85.10 ± 2.97; see
# docs/improvement-handoff-2026-06-13.md). Every SVD-init run in this repo is at
# lr=1e-2 — the lower-LR SVD probe in experiments/ledger/README.md item #5 was
# planned (2026-06-11) but never run.
#
# WHAT THIS RUNS (and what it does NOT)
# -------------------------------------
# Only the MISSING arm: SVD-compensated init at the extra LRs. We already have
# 3-seed VANILLA RoLoRA baselines at these LRs from W&B
# (evidence/wandb_qnli_c50_r4_20260603/figures/proxy_plot_summary.md):
#     RoLoRA lr5e-3 = 81.47 ± 7.23 final / 85.07 ± 0.90 best  (3 seeds)
#     RoLoRA lr1e-2 = 85.10 ± 2.97                             (3 seeds, the control)
#     RoLoRA lr2e-2 = 81.20 ± 9.39 final / 83.23 ± 5.95 best   (3 seeds)
# so do NOT re-run vanilla. After this finishes you have a paired SVD-vs-vanilla
# delta at {5e-3, 1e-2(done), 2e-2}.
#
# VERDICT CRITERION (from the ledger): the "limited improvement" only survives
# if SVD beats BOTH the vanilla control at the same LR AND svd@1e-2. If it
# doesn't, that is itself the reportable answer ("the 0.6 pp does not hold up
# under LR tuning") — which is exactly what the reviewer asked us to check.
#
# Cell: QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds. Proxy evidence,
# NOT RoBERTa-Large Table-1 reproduction.
#
# USAGE
#   # fast harness/venv validation first (~2 min, 2 rounds, seed0 lr5e-3):
#   SMOKE=1 bash scripts/queues/run_svd_lr_sweep_queue.sh
#   # full sweep (default 6 runs ≈ 8 h on this Mac's MPS, ~80 min/run):
#   bash scripts/queues/run_svd_lr_sweep_queue.sh
#   # background + detached so it survives the terminal:
#   nohup bash scripts/queues/run_svd_lr_sweep_queue.sh > results/svd_lr_sweep_queue_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# To widen/narrow: edit LRS and SEEDS below. Adding 0.002 gives a 4th point but
# that is the underfit regime (vanilla lr2e-3 only reached ~56%), so it mostly
# pads the figure rather than testing the improvement.
set -u

cd "$(dirname "$0")/../.."   # repo root (script lives in scripts/queues/), so relative paths resolve

CONFIG="experiments/configs/proxy_qnli_roberta_base_c50_r4_lr1e-2.yaml"
COMMON_TAGS="local,proxy,qnli,roberta-base,c50,r4,rolora,svd_compensated,monitor,lr-sweep"

# --- sweep grid (edit here) -------------------------------------------------
LRS=(0.005 0.02)        # add 0.002 for a 4th (underfit) point if desired
SEEDS=(0 1 2)           # match the existing svd@1e-2 and vanilla seeds
# ---------------------------------------------------------------------------

# numeric lr -> compact label used in the TAG / log filename / W&B name
lr_label() {
  case "$1" in
    0.005) echo "5e-3" ;;
    0.02)  echo "2e-2" ;;
    0.002) echo "2e-3" ;;
    0.01)  echo "1e-2" ;;
    *)     echo "${1//./p}" ;;
  esac
}

run_arm() {
  local lr="$1" seed="$2" rounds="${3:-}" tags="$4"
  local label tag
  label="$(lr_label "$lr")"
  tag="proxy_svd_compensated_c50_r4_lr${label}_seed${seed}"
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') START $tag (lr=$lr rounds=${rounds:-default}) ====="
  unset SLS_PHASE_POLICY
  unset SLS_PHASE_PATTERN
  export SLS_LORA_INIT=svd_compensated
  export SLS_MONITOR=1
  export SLS_DEVICE=mps
  export MODE=rolora
  export TAG="$tag"
  export SEED="$seed"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_ENTITY="${WANDB_ENTITY:-scalable-learning-7}"
  export WANDB_PROJECT="${WANDB_PROJECT:-sls-rolora-repro}"
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qnli_c50_r4_svd_lr_sweep}"
  export WANDB_NAME="$tag"
  export WANDB_TAGS="$tags"

  local overrides=(seed "$seed" train.optimizer.lr "$lr")
  [[ -n "$rounds" ]] && overrides+=(federate.total_round_num "$rounds")

  bash scripts/run_supplement_arm.sh "$CONFIG" "${overrides[@]}"
  local rc=$?
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') END $tag rc=$rc ====="
  return "$rc"
}

# --- SMOKE: one quick 2-round arm to validate venv/harness, then exit -------
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "### SMOKE: SVD lr=5e-3 seed0, 2 rounds only (validation, not a result) ###"
  run_arm 0.005 0 2 "$COMMON_TAGS,smoke"
  rc=$?
  echo "### SMOKE done rc=$rc — if 0, run the full sweep without SMOKE=1 ###"
  exit "$rc"
fi

# --- FULL SWEEP -------------------------------------------------------------
failures=0
for lr in "${LRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_arm "$lr" "$seed" "" "$COMMON_TAGS,seed${seed}" || failures=$((failures+1))
  done
done

echo
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') SVD LR SWEEP DONE failures=$failures ====="
echo
echo "Final-accuracy tables (paired against existing vanilla baselines):"
for lr in "${LRS[@]}"; do
  label="$(lr_label "$lr")"
  echo "--- SVD lr=${label} ---"
  uv run python scripts/summarize_supplement.py \
    --prefix "overnight_proxy_svd_compensated_c50_r4_lr${label}" 2>/dev/null \
    || echo "  (summarize failed; logs are at results/overnight_proxy_svd_compensated_c50_r4_lr${label}_seed*.log)"
done
echo
echo "Compare against: docs/improvement-handoff-2026-06-13.md (svd@1e-2 = 85.70 ± 0.59,"
echo "vanilla@1e-2 = 85.10 ± 2.97) and proxy_plot_summary.md (vanilla@5e-3, @2e-2)."
exit "$failures"

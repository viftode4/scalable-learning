#!/usr/bin/env bash
# Run a supplement config through our patched trainer.
# Usage:
#   bash scripts/smoke_supplement.sh [rolora|lora|ffa_lora|all ...]
#   MODE=all make supplement-smoke
#   CONFIG=experiments/configs/table1_local_pilot.yaml LOG_PREFIX=table1_pilot bash scripts/smoke_supplement.sh all
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SUPP="$REPO/code/harness/rolora-supplement/RoLoRA-code"
VENV="$SUPP/.venv-supplement"
CONFIG=${CONFIG:-"$REPO/experiments/configs/smoke_supplement.yaml"}
# Auto-derive log prefix from the config basename when not explicitly set.
# Lets per-cell sbatch jobs land at results/<config_stem>_<mode>[_seedN].log without
# every caller having to thread LOG_PREFIX through env.
LOG_PREFIX=${LOG_PREFIX:-$(basename "$CONFIG" .yaml)}
RESULTS="$REPO/results"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "error: supplement venv missing at $VENV"
    echo "  run: make install-supplement"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "error: missing smoke config: $CONFIG"
    exit 1
fi

if [[ ! -f "$SUPP/sst2/qnli.json" ]]; then
    echo "[prep] creating supplement QNLI JSON"
    (cd "$SUPP/sst2" && "$VENV/bin/python" qnli2json.py)
fi

mkdir -p "$RESULTS"

modes=("${@:-${MODE:-rolora}}")
if [[ "${modes[0]}" == "all" ]]; then
    modes=(rolora lora ffa_lora)
fi

for mode in "${modes[@]}"; do
    case "$mode" in
        rolora|lora|ffa_lora) ;;
        *)
            echo "error: unknown mode '$mode' (expected rolora, lora, ffa_lora, or all)"
            exit 2
            ;;
    esac

    seed_suffix=""
    seed_override=()
    if [[ -n "${SEED:-}" ]]; then
        seed_suffix="_seed${SEED}"
        seed_override=(seed "$SEED")
    fi

    log="$RESULTS/${LOG_PREFIX}_${mode}${seed_suffix}.log"
    echo "[smoke] $mode${seed_suffix:+ seed=$SEED} -> $log"

    if {
        echo "# repo: $REPO"
        echo "# git_sha: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "# config: $CONFIG"
        echo "# mode: $mode"
        echo "# seed: ${SEED:-default}"
        echo "# started_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        cd "$REPO"
        NO_COLOR=1 SLS_ALTERNATION_MODE="$mode" \
            "$VENV/bin/python" scripts/run_supplement.py \
            --cfg "$CONFIG" "${seed_override[@]}"
    } >"$log" 2>&1; then
        marker="$(grep -F "[sls-rolora]" "$log" | tail -1 | perl -pe 's/\e\[[0-9;]*m//g' || true)"
        final="$(grep -E "Results_raw|Results_avg|Results_weighted_avg" "$log" | tail -1 | perl -pe 's/\e\[[0-9;]*m//g' || true)"
        if [[ -z "$marker" ]]; then
            echo "error: smoke run finished but patch marker was missing in $log"
            tail -80 "$log"
            exit 3
        fi
        echo "  $marker"
        if [[ -n "$final" ]]; then
            echo "  final: $final"
        fi
        echo "[ok] $mode"
    else
        echo "[fail] $mode; tail of $log:"
        tail -80 "$log"
        exit 1
    fi
done

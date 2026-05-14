#!/usr/bin/env bash
# Run the tiny OpenReview-supplement smoke config through our patched trainer.
# Usage:
#   bash scripts/smoke_supplement.sh [rolora|lora|ffa_lora|all ...]
#   MODE=all make supplement-smoke
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SUPP="$REPO/code/harness/rolora-supplement/RoLoRA-code"
VENV="$SUPP/.venv-supplement"
CONFIG="$REPO/experiments/configs/smoke_supplement.yaml"
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

    log="$RESULTS/smoke_${mode}.log"
    echo "[smoke] $mode -> $log"

    if (
        cd "$REPO"
        NO_COLOR=1 SLS_ALTERNATION_MODE="$mode" \
            "$VENV/bin/python" scripts/run_supplement.py \
            --cfg "$CONFIG"
    ) >"$log" 2>&1; then
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

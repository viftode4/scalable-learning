#!/usr/bin/env bash
# Unpack the OpenReview supplementary zip for RoLoRA into code/harness/rolora-supplement/.
# Usage:
#   bash scripts/extract_supplement.sh [path/to/zip]
set -euo pipefail

DEST="code/harness/rolora-supplement"

if [[ $# -gt 0 ]]; then
    ZIP_PATH="$1"
else
    candidates=(
        "$HOME/Downloads/5662_Robust_Federated_Finetuni_Supplementary Material.zip"
        "$HOME/Downloads/rolora-supplement.zip"
        "$HOME/Downloads/supplement.zip"
        "$HOME/Downloads/code.zip"
    )
    ZIP_PATH=""
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            ZIP_PATH="$candidate"
            break
        fi
    done
fi

if [[ -z "${ZIP_PATH:-}" || ! -f "$ZIP_PATH" ]]; then
    echo "error: supplement zip not found."
    echo "  expected one of:"
    echo "    ~/Downloads/5662_Robust_Federated_Finetuni_Supplementary Material.zip"
    echo "    ~/Downloads/rolora-supplement.zip"
    echo "  or pass an explicit path: bash scripts/extract_supplement.sh /path/to/zip"
    exit 1
fi

mkdir -p "$DEST"
echo "extracting $ZIP_PATH -> $DEST"
unzip -q -o "$ZIP_PATH" -d "$DEST"
find "$DEST" -name '.DS_Store' -delete
rm -rf "$DEST/__MACOSX"

echo
echo "contents:"
find "$DEST" -maxdepth 3 -type f | head -50

echo
echo "sha256 of zip:"
shasum -a 256 "$ZIP_PATH" | awk '{print $1}'

PY_COUNT=$(find "$DEST" -name '*.py' | wc -l | tr -d ' ')
echo
echo "python files found: $PY_COUNT"
if [[ "$PY_COUNT" -eq 0 ]]; then
    echo "warning: no .py files in the extracted supplement."
    echo "  the supplement may be misnamed or corrupt — re-download from OpenReview."
    exit 2
fi

echo "ok."

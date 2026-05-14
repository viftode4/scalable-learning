#!/usr/bin/env bash
# Unpack the OpenReview supplementary zip for RoLoRA into code/harness/rolora-supplement/.
# Usage:
#   bash scripts/extract_supplement.sh [path/to/zip]   # default: ~/Downloads/rolora-supplement.zip
set -euo pipefail

ZIP_PATH="${1:-$HOME/Downloads/rolora-supplement.zip}"
DEST="code/harness/rolora-supplement"

if [[ ! -f "$ZIP_PATH" ]]; then
    echo "error: zip not found at $ZIP_PATH"
    echo "  follow docs/setup/openreview-supplement.md to download it first."
    exit 1
fi

mkdir -p "$DEST"
echo "extracting $ZIP_PATH -> $DEST"
unzip -q -o "$ZIP_PATH" -d "$DEST"

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

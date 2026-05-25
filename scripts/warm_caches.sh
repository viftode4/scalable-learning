#!/usr/bin/env bash
#
# Warm caches on the LOGIN NODE (DelftBlue compute nodes have no outbound
# internet access — `huggingface.co` returns [Errno 101] Network is
# unreachable from gpu* nodes, so the model must be fetched here).
#
# This script is conceptually equivalent to running wget for many files:
# it uses huggingface_hub.snapshot_download (file fetch only, no model
# load) and datasets.load_dataset (small dataset download + JSON write).
# Memory footprint is small (~200–500 MB peak); runtime is ~5 min,
# mostly waiting on network.
#
# Usage (one-time, on the DelftBlue login node):
#   bash scripts/warm_caches.sh
#
# Verify after it returns:
#   [ -d "$HF_HOME/hub/models--roberta-large" ] && \
#   [ -s code/harness/rolora-supplement/RoLoRA-code/sst2/qnli.json ] && \
#   echo ok

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Use the supplement venv so we match the versions training will use.
SUPP_PY="$REPO/code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python"
[ -x "$SUPP_PY" ] || {
    echo "FAIL: supplement venv missing at $SUPP_PY"
    echo "      Run 'bash scripts/install_supplement.sh' first."
    exit 1
}

# HF cache must be on scratch so sbatch jobs can read it.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME = $HF_HOME"

# ---------------------------------------------------------------------------
# [1/2] Pre-download roberta-large weights + tokenizer.
#
# snapshot_download writes files to $HF_HOME/hub/ but never instantiates the
# model — so RAM footprint is dominated by the Python interpreter, NOT the
# model weights. This keeps it light enough for the login node.
# ---------------------------------------------------------------------------
echo "[1/2] Pre-downloading roberta-large into $HF_HOME ..."
"$SUPP_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='roberta-large')
print('roberta-large snapshot path:', path)
# Cheap presence check — make sure we have what from_pretrained will need.
for fname in ['config.json', 'pytorch_model.bin']:
    fpath = os.path.join(path, fname)
    assert os.path.exists(fpath), f'snapshot missing {fpath}'
print('roberta-large cache ready.')
PY

# ---------------------------------------------------------------------------
# [2/2] Prepare sst2/qnli.json (the supplement reads this directly).
# qnli2json.py uses datasets.load_dataset("glue", "qnli") + JSON write.
# ---------------------------------------------------------------------------
QNLI_JSON="$REPO/code/harness/rolora-supplement/RoLoRA-code/sst2/qnli.json"
if [ -s "$QNLI_JSON" ]; then
    echo "[2/2] $QNLI_JSON already exists; skipping prep."
else
    echo "[2/2] Preparing $QNLI_JSON ..."
    (
        cd "$REPO/code/harness/rolora-supplement/RoLoRA-code/sst2"
        "$SUPP_PY" qnli2json.py
    )
fi

echo
echo "Warm caches complete."
ls -lh "$HF_HOME/hub/models--roberta-large/" 2>/dev/null || true
ls -lh "$QNLI_JSON" 2>/dev/null || true

#!/usr/bin/env bash
# Install the OpenReview RoLoRA supplement in an isolated Python 3.9 venv next to
# the extracted supplement, apply our sls-rolora patch, and import-test the result.
#
# Why isolated: the supplement pins numpy<1.23 / protobuf==3.19.4 / peft==0.3.0 /
# transformers==4.29.2 — incompatible with our main env (peft==0.10.0).
#
# Why no `--editable`: the supplement's setup.py uses a build backend that does
# not support PEP 660 editable installs.
#
# Usage:
#   bash scripts/install_supplement.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SUPP="$REPO/code/harness/rolora-supplement/RoLoRA-code"
VENV="$SUPP/.venv-supplement"
PATCH="$REPO/code/harness/rolora-supplement.patch"

if [[ ! -d "$SUPP" ]]; then
    echo "error: supplement not extracted at $SUPP"
    echo "  run scripts/extract_supplement.sh first"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not on PATH; install from https://github.com/astral-sh/uv"
    exit 1
fi

# 1. Python 3.9 venv (supplement README requirement).
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python 3.9 "$VENV"
    "$VENV/bin/python" -m ensurepip
    "$VENV/bin/python" -m pip install --upgrade 'pip>=23' 'setuptools<70' wheel
fi

# 2. Apply our trainer patch (idempotent — patch checks if already applied).
if grep -q SLS_ALTERNATION_MODE "$SUPP/federatedscope/llm/trainer/trainer.py"; then
    echo "[skip] sls-rolora patch already applied"
else
    echo "[apply] sls-rolora trainer patch"
    (cd "$SUPP" && patch -p1 < "$PATCH")
fi

# 3. Install federatedscope core (no editable; old setup.py doesn't support PEP 660).
"$VENV/bin/python" -m pip install --no-build-isolation "$SUPP"

# 4. Install the LLM-extra pins.
# Note: `datasets` is required by the dataset-prep scripts in sst2/ but is NOT in
# the supplement's [llm] extras (it's only in [app]). We add it explicitly.
"$VENV/bin/python" -m pip install --no-build-isolation \
    'torch>=2.0,<2.4' \
    'transformers==4.29.2' \
    'tokenizers==0.13.3' \
    'accelerate==0.20.3' \
    'peft==0.3.0' \
    'sentencepiece==0.1.99' \
    'datasets'

# 5. Import sanity check.
"$VENV/bin/python" - <<'PY'
import federatedscope, torch, transformers, peft, accelerate
print("OK:",
      "federatedscope", federatedscope.__version__,
      "| torch", torch.__version__,
      "| transformers", transformers.__version__,
      "| peft", peft.__version__,
      "| accelerate", accelerate.__version__)
PY

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Run an experiment:"
echo "  SLS_ALTERNATION_MODE=rolora \\"
echo "  $VENV/bin/python $SUPP/federatedscope/main.py \\"
echo "    --cfg $SUPP/federatedscope/llm/baseline/test_glue.yaml"

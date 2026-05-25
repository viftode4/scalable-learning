#!/usr/bin/env bash
#
# Push the local repo to DelftBlue via rsync over SSH.
#
# Assumes you are on TU Delft Wi-Fi or connected via EduVPN (no bastion hop).
#
# Usage:
#   bash scripts/sync_to_delftblue.sh <netid>
# or:
#   NETID=<netid> bash scripts/sync_to_delftblue.sh
#
# By default this pushes to ~/scalable-learning/ on the login node. Override
# with REMOTE_PATH=~/some/other/dir if you want a different target.

set -euo pipefail

NETID="${1:-${NETID:-}}"
if [[ -z "$NETID" ]]; then
  echo "usage: $0 <netid>   (or set NETID env var)" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-login.delftblue.tudelft.nl}"
REMOTE_PATH="${REMOTE_PATH:-~/scalable-learning/}"

# ServerAlive keeps the connection from dying mid-transfer on flaky Wi-Fi.
SSH_CMD="ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3"

echo "Syncing $REPO -> ${NETID}@${REMOTE_HOST}:${REMOTE_PATH}"

# Exclusions:
#   - .git/, .venv/, __pycache__/, *.pyc, IDE/macOS junk     → useless on cluster
#   - .ruff_cache/ / .mypy_cache/ / .pytest_cache/           → regenerated locally
#   - wandb/, slurm_logs/, logs/, runs/, mlruns/, exp/       → cluster-generated outputs
#   - results/                                               → laptop-side experiment outputs
#   - data/                                                  → big binary data, rebuilt on cluster
#   - .venv-supplement/                                      → Python 3.9 venv rebuilt by install_supplement.sh
#   - *.ckpt / *.pt                                          → model checkpoints; transfer separately if needed
rsync -avzh --partial --progress \
  -e "$SSH_CMD" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='wandb/' \
  --exclude='slurm_logs/' \
  --exclude='logs/' \
  --exclude='runs/' \
  --exclude='mlruns/' \
  --exclude='exp/' \
  --exclude='results/' \
  --exclude='data/' \
  --exclude='code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/' \
  --exclude='*.ckpt' \
  --exclude='*.pt' \
  "$REPO/" "${NETID}@${REMOTE_HOST}:${REMOTE_PATH}"

echo
echo "Done. Next steps on the cluster:"
echo "  ssh ${NETID}@${REMOTE_HOST}"
echo "  cd ~/scalable-learning"
echo "  # follow docs/setup/delftblue.md from step 2 onwards"

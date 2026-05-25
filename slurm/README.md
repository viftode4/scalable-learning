# Slurm templates — DelftBlue

Real, work-now sbatch scripts for the QNLI reproduction sweep on
`gpu-a100-small`. Modelled on the reference DelftBlue template in
`sbatch_examples/run_paper_experiment_48actors.sbatch`, sized for our
single-process supplement workload.

## Currently in the repo

### One-time setup (runs on the login node, NOT slurm)

| File | What it does |
|---|---|
| `../scripts/warm_caches.sh` | Pre-downloads `roberta-large` to `HF_HOME=/scratch/$USER/hf-cache` and prepares `sst2/qnli.json`. Runs on the login node because DelftBlue compute nodes have no outbound internet (huggingface.co is unreachable). Uses `huggingface_hub.snapshot_download` so it's pure file fetching — no model load, no heavy compute. |

### Cell C2 worked example (3 training files)

| File | Method | Config |
|---|---|---|
| `repro_qnli_c20_r4_lora.sbatch` | LoRA | `experiments/configs/repro_qnli_c20_r4.yaml` |
| `repro_qnli_c20_r4_ffa_lora.sbatch` | FFA-LoRA | `experiments/configs/repro_qnli_c20_r4.yaml` |
| `repro_qnli_c20_r4_rolora.sbatch` | RoLoRA | `experiments/configs/repro_qnli_c20_r4.yaml` |

All three target paper-cell C2 (20 clients, RoBERTa-Large, LoRA rank 4) on
`gpu-a100-small`. They share identical resource asks (`--time=03:59:59`,
`--cpus-per-task=2`, `--gpus-per-task=1`, `--mem-per-cpu=8000M` (~16G total; `gpu-a100-small` caps both `cpus-per-task` at 2 and `mem-per-cpu` at 8000 MB), account
`education-eemcs-msc-dsait`); they differ only in `SLS_ALTERNATION_MODE`,
`--job-name`, and the wandb `WANDB_NAME` / `WANDB_TAGS`.

C1, C3, C4 sbatch files are not yet authored — they are deferred until C2
has actually completed cleanly on the cluster. The C2 template is the
copy-and-edit source for the remaining 9 (cell × mode) variants.

## First-time setup — the dum-dum checklist

Run these in order, top-to-bottom. Each step has a `verify` line — if it
prints anything other than `ok`, stop and fix it before continuing. None
of these run real Python on the login node (DelftBlue policy); the heavy
download lives in step 8 as a slurm job.

For the long-form rationale per step, see
[`docs/setup/delftblue.md`](../docs/setup/delftblue.md). This list is what
you actually type.

```bash
# === 0. ON YOUR LAPTOP, BEFORE SSH ==========================================
# Push the repo to the cluster (idempotent; rerun after any local edit):
bash scripts/sync_to_delftblue.sh <netid>
```

```bash
# === 1. SSH IN ==============================================================
ssh <netid>@login.delftblue.tudelft.nl
cd ~/scalable-learning
# verify: should see Makefile, slurm/, experiments/, code/
ls Makefile slurm/ experiments/ code/ >/dev/null && echo ok
```

```bash
# === 2. uv on PATH (skip if `command -v uv` already prints a path) ==========
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
# verify:
command -v uv >/dev/null && echo ok
```

```bash
# === 3. Caches on scratch (compatible with the other-project setup) =========
# This step ONLY appends NEW lines to ~/.bashrc. It never overwrites or
# touches an existing UV_CACHE_DIR / HF_HOME line.
#
# If you already ran the other project's RUN_ON_DELFTBLUE.md, your bashrc
# already exports UV_CACHE_DIR — the grep below finds it and skips. We just
# add HF_HOME (which the other project did not set).
grep -qE '^[[:space:]]*export UV_CACHE_DIR=' ~/.bashrc || {
    mkdir -p /scratch/$USER/uv-cache
    echo 'export UV_CACHE_DIR=/scratch/$USER/uv-cache' >> ~/.bashrc
}
grep -qE '^[[:space:]]*export HF_HOME=' ~/.bashrc || {
    mkdir -p /scratch/$USER/hf-cache
    echo 'export HF_HOME=/scratch/$USER/hf-cache' >> ~/.bashrc
}
source ~/.bashrc
# verify:
[ -n "$UV_CACHE_DIR" ] && [ -n "$HF_HOME" ] && echo ok
```

> ⚠️ **NFS hang on first `import` — read before steps 4 and 5.**
>
> DelftBlue's `/home` is NFS-mounted. The very first time you `import torch`
> (or any heavy package) inside a freshly installed venv, Python writes a
> `.pyc` bytecode cache for *every* module it touches — hundreds of small
> writes on slow NFS. That can take **5–10 minutes** the first time and
> looks frozen but isn't. **Do not Ctrl-C.** Once warmed, subsequent
> imports are sub-second.
>
> The verify lines below avoid triggering this in step 4 (we trust
> `uv sync`'s exit code). Step 5's `install_supplement.sh` runs its own
> import sanity check at the end — that one *will* be slow the first
> time; let it finish.

```bash
# === 4. Main project venv (Python 3.11) =====================================
uv sync
# verify (cheap — no heavy import; we trust uv sync's exit code):
.venv/bin/python --version \
  && ls -d .venv/lib*/python3.11/site-packages/torch >/dev/null \
  && echo ok
```

```bash
# === 5. Supplement venv (Python 3.9; isolated from main .venv) ==============
# First run will be slow (5–10 min) — the script ends with an import sanity
# check that warms the .pyc cache for federatedscope/torch/transformers/
# peft/accelerate/wandb. DO NOT Ctrl-C; if you see no output for several
# minutes the bytecode cache is being written.
bash scripts/install_supplement.sh
# verify (script prints its own "OK: federatedscope ... | torch ..." line at
# the very end if all imports succeeded; this is just a final confirmation):
code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python --version \
  && ls -d code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/lib/python3.9/site-packages/federatedscope >/dev/null \
  && ls -d code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/lib/python3.9/site-packages/wandb >/dev/null \
  && echo ok
```

```bash
# === 6. Slurm log directory (slurm silently drops output without this) ======
mkdir -p slurm_logs
# verify:
[ -d slurm_logs ] && echo ok
```

```bash
# === 7. Wandb credentials (.env is NOT synced from your laptop) =============
# Get the key from https://wandb.ai/authorize
test -f .env || cat > .env <<EOF
WANDB_API_KEY=<paste-your-key-here>
EOF
chmod 600 .env
# verify:
grep -q '^WANDB_API_KEY=..' .env && echo ok
```

```bash
# === 8. Warm caches on the LOGIN NODE (~5 min, mostly network wait) =========
# DelftBlue compute nodes have NO outbound internet — huggingface.co
# returns [Errno 101] Network is unreachable from gpu* nodes. So the model
# + dataset MUST be downloaded on the login node. The script only fetches
# files (no model load), so memory footprint stays small.
#
# DO NOT Ctrl-C this — it's mostly waiting on HF Hub bandwidth.
bash scripts/warm_caches.sh
# verify:
[ -d /scratch/$USER/hf-cache/hub/models--roberta-large ] \
  && [ -s code/harness/rolora-supplement/RoLoRA-code/sst2/qnli.json ] \
  && echo ok
```

```bash
# === 9. (Final pre-flight check) ============================================
# All-in-one verification. Run this BEFORE submitting the first training job.
{
  command -v uv >/dev/null || { echo FAIL uv; exit 1; }
  [ -x .venv/bin/python ] || { echo FAIL .venv; exit 1; }
  [ -x code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python ] || { echo FAIL .venv-supplement; exit 1; }
  [ -d slurm_logs ] || { echo FAIL slurm_logs; exit 1; }
  [ -s .env ] && grep -q WANDB_API_KEY .env || { echo FAIL .env; exit 1; }
  [ -d /scratch/$USER/hf-cache/hub/models--roberta-large ] || { echo FAIL hf-cache; exit 1; }
  [ -s code/harness/rolora-supplement/RoLoRA-code/sst2/qnli.json ] || { echo FAIL qnli.json; exit 1; }
  echo "All prerequisites OK — ready to submit training jobs."
}
```

After step 9 prints `All prerequisites OK`, you're done with one-time
setup. Subsequent re-syncs from your laptop only need step 0 (laptop
rsync); the cluster-side state is already in place.

## Submitting jobs

### First-time order

```bash
# One-time login-node script (~5 min, mostly network wait):
bash scripts/warm_caches.sh
# DO NOT Ctrl-C — it's just downloading files. Once HF_HOME/hub/models--roberta-large
# exists and sst2/qnli.json is non-empty, you're done and can submit training.
```

### Training submissions

```bash
# Seed 0 (default):
sbatch slurm/repro_qnli_c20_r4_rolora.sbatch

# Other seeds — same sbatch file, SEED env var overrides:
SEED=1 sbatch slurm/repro_qnli_c20_r4_rolora.sbatch
SEED=2 sbatch slurm/repro_qnli_c20_r4_rolora.sbatch
```

Each invocation produces:
- One slurm log: `slurm_logs/sls-c20-r4-<mode>-<jobid>.out`
- One supplement log: `results/repro_qnli_c20_r4_<mode>_seed<N>.log` (auto-named from config basename)
- One FederatedScope per-run directory: `exp/<long_run_name>/sub_exp_<ts>/`
  containing `eval_results.log` with per-round eval metrics (source-of-truth
  for the post-hoc Figure-3 convergence plotter that's not yet built)
- One wandb run, grouped under `qnli_c20_r4` in project `sls-rolora-repro`

## What's NOT in the repo yet (deferred)

- `repro_qnli_c{3,50}_r{4,8}_{lora,ffa_lora,rolora}.sbatch` (9 more files,
  one per remaining cell × mode).
- `scripts/submit_repro_qnli.sh` — bash orchestrator for the full 36-job
  sweep that records `(jobid, cell, mode, seed)` tuples for ledger
  traceability.
- `scripts/aggregate_seeds.py` — mean ± std across seeds, producing the
  Table-1-shape grid.
- `scripts/plot_convergence_curves.py` — reads `exp/*/sub_exp_*/eval_results.log`
  to produce the Figure-3-style convergence panel.

These all land in a second pass after the C2 worked example actually runs
on the cluster.

## Gotchas

- **`mkdir -p slurm_logs` is required** — without it, slurm silently drops
  job output.
- **`uv sync` and `bash scripts/install_supplement.sh` must run on the login
  node**, not inside the job. Compute nodes generally have no outbound
  network for package installs.
- **`gpu-a100-small` has a hard 4h wall-time cap** per MIG slice. The
  account ceiling is higher (24h) but the slice cap is what binds. Our
  scripts request `03:59:59` (max under cap); C2 jobs are estimated at
  ~30 min so this is comfortable. The 50-client cells (~75-80 min) also
  fit; if they ever exceed 4h we'd move them to `gpu-a100` (full A100,
  24h account cap).
- **Wandb offline fallback**: if compute nodes have no outbound network,
  wandb buffers to `wandb/offline-run-*` and you `wandb sync` later.
  The local `results/*.log` and `exp/*/sub_exp_*/eval_results.log` files
  are authoritative regardless of wandb state.

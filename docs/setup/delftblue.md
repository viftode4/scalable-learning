# Run on DelftBlue — step-by-step

This is the dum-dum runbook. Follow it top-to-bottom for the **first**
submission of a fresh repo; thereafter only sections 4–6 matter per job.
Every step has a verification command — if the verification fails, do not
proceed.

If you've already set up DelftBlue for another project (e.g. uv already
installed, `UV_CACHE_DIR` already pointing at `/scratch/$USER/...`, `.env`
already populated), the existing setup is reused — none of the steps below
overwrite it. The supplement-venv install is the only new thing.

## Login-node policy (read this before touching anything)

DelftBlue login nodes are shared and **heavily restricted**. The rules
this runbook follows:

- ✅ **Allowed on the login node**: `uv sync`, `pip install`, `bash scripts/install_supplement.sh`, file edits, `git`, `rsync`, `sbatch`, `squeue`, `tail`, `cat`, env var exports, light `python -c "import x; print(x.__version__)"` sanity checks.
- ❌ **Not allowed on the login node**: model downloads, `from_pretrained(...)`, dataset prep, training runs, anything that touches a GPU, any Python process that allocates more than a few hundred MB or runs longer than ~10 seconds.
- ❌ **No interactive compute work**. No `srun --pty bash`, no GPU sessions. Compute work is ALWAYS submitted via `sbatch`.
- ✅ **Compute nodes** (via `sbatch`): real training, GPU work, anything with `from_pretrained`. **No outbound internet** — `huggingface.co` is unreachable from `gpu*` nodes; all model + dataset downloads must already be cached on scratch when the job starts. Wandb's specific endpoint *does* appear to be reachable (the user's other project uses wandb from compute nodes successfully), but if your training fails with a wandb network error we'll switch to `WANDB_MODE=offline` and `wandb sync` after the run.

The runbook's "heavy" steps (model + dataset warm-cache) therefore run
via `scripts/warm_caches.sh` on the login node — using
`huggingface_hub.snapshot_download` (file fetch only, no model load),
which is light enough to fit the login-node policy.

## Cluster configuration (constants)

| Knob | Value |
|---|---|
| Account | `education-eemcs-msc-dsait` |
| Modules loaded by sbatch | `2024r1`, `cuda` |
| Primary partition | `gpu-a100-small` (10 GB MIG, ≤4h cap) |
| Backup partition | `gpu-a100` (full 80 GB, 24h account cap) |
| Login host | `login.delftblue.tudelft.nl` (EduVPN if off-campus) |
| Wandb entity | `scalable-learning-7` |
| Wandb project | `sls-rolora-repro` |
| Repo home on cluster | `~/scalable-learning` |
| Scratch root | `/scratch/$USER` |

## 1. Push the repo from your laptop

```bash
# On your laptop, from inside the repo:
bash scripts/sync_to_delftblue.sh <netid>
```

This rsyncs to `~/scalable-learning/` on the login node, excluding
`.venv/`, `.venv-supplement/`, `exp/`, `results/`, `slurm_logs/`, `wandb/`,
`*.ckpt`, `*.pt`, IDE / cache junk, and anything else that's regenerated
or too big.

After it finishes:

```bash
ssh <netid>@login.delftblue.tudelft.nl
cd ~/scalable-learning
ls    # sanity: should see Makefile, slurm/, experiments/, code/, etc.
```

## 2. One-time setup on the login node

Skip whichever pieces you already have from another project. Each step has
a `verify` line — if verify fails, run the install line above it.

### 2.1 uv

```bash
# Install (skip if `command -v uv` already returns a path):
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# verify:
command -v uv && uv --version
```

### 2.2 Redirect uv cache to scratch (saves /home space)

Skip if `echo $UV_CACHE_DIR` already prints `/scratch/<netid>/...`.

```bash
mkdir -p /scratch/$USER/uv-cache
echo 'export UV_CACHE_DIR=/scratch/$USER/uv-cache' >> ~/.bashrc
source ~/.bashrc

# verify:
echo "$UV_CACHE_DIR"   # should print /scratch/<netid>/uv-cache
```

### 2.3 Main project venv (Python 3.11)

```bash
cd ~/scalable-learning
uv sync

# verify:
.venv/bin/python --version       # 3.11.x
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
```

### 2.4 Supplement venv (Python 3.9, isolated from main)

The supplement (FederatedScope fork) pins `numpy<1.23`, `peft==0.3.0`,
`transformers==4.29.2` — incompatible with our main env. It lives in its
own venv inside the vendored supplement directory.

```bash
cd ~/scalable-learning
bash scripts/install_supplement.sh

# verify:
code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python --version  # 3.9.x
code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python -c "import wandb, torch, federatedscope; print('ok')"
```

### 2.5 Slurm log directory (required — slurm silently drops output without it)

```bash
mkdir -p ~/scalable-learning/slurm_logs

# verify:
[ -d ~/scalable-learning/slurm_logs ] && echo "ok"
```

### 2.6 Wandb credentials

Get your API key from <https://wandb.ai/authorize>.

If you already have a working `~/scalable-learning/.env` from before, skip
this — `scripts/sync_to_delftblue.sh` does **not** sync `.env` (it's in
`.gitignore` and the rsync excludes match git-side). Create on the cluster
directly:

```bash
cd ~/scalable-learning
test -f .env || cat > .env <<EOF
WANDB_API_KEY=<paste-key-here>
EOF
chmod 600 .env

# verify:
[ -s .env ] && grep -q WANDB_API_KEY .env && echo "ok"
```

### 2.7 Set the HuggingFace cache path on scratch

DelftBlue policy: **do not** run heavy Python (model loading, dataset
download) on the login node. We point `HF_HOME` at scratch here; the
actual download happens in section 2.8 via slurm.

```bash
# Point HF cache at scratch (same path the sbatch scripts use):
mkdir -p /scratch/$USER/hf-cache
echo 'export HF_HOME=/scratch/$USER/hf-cache' >> ~/.bashrc
source ~/.bashrc

# verify:
echo "$HF_HOME"   # should print /scratch/<netid>/hf-cache
```

### 2.8 Warm caches on the login node (one-time, ~5 min)

DelftBlue compute nodes have **no outbound internet** —
`huggingface.co` returns `[Errno 101] Network is unreachable` from
`gpu*` nodes. So the model weights (~1.4 GB) and the QNLI dataset must
be downloaded on the **login node** before any training submission.

The script uses `huggingface_hub.snapshot_download`, which is pure file
fetching (no model instantiation, ~200 MB RAM peak). It's morally
equivalent to running `wget` for many files — fits the "install
script or two" policy.

```bash
cd ~/scalable-learning
bash scripts/warm_caches.sh
```

This is mostly network wait. **Do not Ctrl-C** — let it finish.

```bash
# verify:
[ -d /scratch/$USER/hf-cache/hub/models--roberta-large ] \
  && [ -s code/harness/rolora-supplement/RoLoRA-code/sst2/qnli.json ] \
  && echo "ok"
```

If the script printed `FAIL: supplement venv missing` you forgot
section 2.4 — go back and run `bash scripts/install_supplement.sh`
first, then re-run.

## 3. Full setup verification (single command)

Run this once after section 2; it will refuse to proceed if anything is
wrong:

```bash
cd ~/scalable-learning
{
  command -v uv >/dev/null || { echo "FAIL: uv not on PATH"; exit 1; }
  [ -x .venv/bin/python ] || { echo "FAIL: main .venv missing — run 'uv sync'"; exit 1; }
  [ -x code/harness/rolora-supplement/RoLoRA-code/.venv-supplement/bin/python ] || \
    { echo "FAIL: supplement venv missing — run scripts/install_supplement.sh"; exit 1; }
  [ -d slurm_logs ] || { echo "FAIL: slurm_logs/ missing"; exit 1; }
  [ -s .env ] && grep -q WANDB_API_KEY .env || { echo "FAIL: .env / WANDB_API_KEY missing"; exit 1; }
  [ -d /scratch/$USER/hf-cache/hub/models--roberta-large ] || \
    { echo "FAIL: roberta-large not pre-cached — run 'bash scripts/warm_caches.sh' (step 2.8)"; exit 1; }
  [ -s code/harness/rolora-supplement/RoLoRA-code/sst2/qnli.json ] || \
    { echo "FAIL: qnli.json not generated — run 'bash scripts/warm_caches.sh' (step 2.8)"; exit 1; }
  echo "All cluster prerequisites OK."
}
```

## 4. Submit a job (the actual work)

**You must submit from `~/scalable-learning` on the login node.** The sbatch
scripts use `cd "$SLURM_SUBMIT_DIR"` internally, so as long as you ran
`sbatch` from the repo root, all relative paths in the job will resolve.

```bash
cd ~/scalable-learning

# RoLoRA, seed 0:
sbatch slurm/repro_qnli_c20_r4_rolora.sbatch

# LoRA, seed 0:
sbatch slurm/repro_qnli_c20_r4_lora.sbatch

# FFA-LoRA, seed 0:
sbatch slurm/repro_qnli_c20_r4_ffa_lora.sbatch

# Any of the above with a different seed:
SEED=1 sbatch slurm/repro_qnli_c20_r4_rolora.sbatch
SEED=2 sbatch slurm/repro_qnli_c20_r4_rolora.sbatch
```

Each invocation returns a job id (e.g. `Submitted batch job 1234567`).

## 5. Watch a running / queued job

```bash
# Queue + run state for your jobs (PD=pending, R=running, CG=completing):
squeue -u $USER

# Live tail of the slurm job log (replace <jobname> and <jobid>):
tail -f slurm_logs/sls-c20-r4-rolora-<jobid>.out

# After the job exits, the supplement log (auto-named from config):
tail results/repro_qnli_c20_r4_rolora_seed0.log

# Per-round eval trajectory (the source for Figure-3-style curves):
tail exp/<run_dir>/sub_exp_<timestamp>/eval_results.log
```

Wandb dashboard (real-time during the run):
- Project URL: <https://wandb.ai/scalable-learning-7/sls-rolora-repro>
- Filter on group `qnli_c20_r4` to see the C2 cell across all methods/seeds.

## 6. Cancel a job

```bash
scancel <jobid>           # single job
scancel -u $USER          # all your jobs (use with care)
```

## Gotchas (read before your first submission)

1. **Submit from the repo root.** `cd ~/scalable-learning && sbatch slurm/...`. Not from `~`, not from `slurm/`. The sbatch file does `cd "$SLURM_SUBMIT_DIR"` internally so the working dir matches submission dir.
2. **`mkdir -p slurm_logs` is required once.** Without it slurm silently drops job stdout/stderr.
3. **`uv sync` and `scripts/install_supplement.sh` only on the login node.** Compute nodes typically have no outbound network for package installs.
4. **`roberta-large` must be pre-cached.** See section 2.7. The cluster job will fail at model load if it isn't there.
5. **Education account caps at 24h wall time**, `gpu-a100-small` partition caps at 4h per MIG slice. Our sbatch scripts request 03:59:59 (max under the 4h cap). C2 jobs are estimated at ~30 min so this is comfortable.
6. **`.env` is NOT synced from your laptop.** It is in `.gitignore` and the rsync exclusion list. Create it directly on the cluster.
7. **Wandb offline fallback**: if compute nodes block outbound HTTPS, wandb buffers runs into `wandb/offline-run-*` directories. After the job finishes, `wandb sync wandb/offline-run-*` on the login node uploads them. The local `results/*.log` and `exp/*/sub_exp_*/eval_results.log` files remain authoritative regardless.
8. **Backspace broken over SSH from some terminals?** Set `TERM=xterm-256color` before `ssh`.
9. **If you're not on the DSAIT account**, run `sacctmgr list user $USER withassoc format='user%-20,account%-45'` to find your account name, then edit `--account=` in the three sbatch files.

## What's intentionally NOT here yet

- Sbatch files for cells C1 (3 clients), C3 (50 clients, r=4), C4 (50 clients, r=8). Add by copying any of the C2 sbatch files and changing `CONFIG_PATH`, `--job-name`, `WANDB_RUN_GROUP`, `WANDB_TAGS`. Authoring those is on the to-do until C2 has actually completed on the cluster.
- A submission orchestrator that fires all 36 jobs in one go and records `(jobid, cell, mode, seed)` for ledger traceability.
- A post-hoc seed-aggregator and convergence-curve plotter.

See `slurm/README.md` for the deferred-list summary and `docs/experiment-matrix.md` for the full sweep shape.

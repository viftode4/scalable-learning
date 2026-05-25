# Agent / Assistant Guidance — `scalable-learning`

## What this repo is
TU Delft **CS 4725** research seminar — 9 course weeks, currently in **week 3** of 9. Team of 3 (Iftode / Popovici / Zele) reproducing RoLoRA (Chen et al., NeurIPS 2025) and proposing improvements. Project officially starts week 4; proposal already submitted 12 May 2026; hard end is week 9 (final report + final presentation, only graded deliverables).

See `README.md` for orientation and `docs/research/` for canonical source docs.

## Hard constraints
- **No unrelated prior-project framing** — keep the project centered on RoLoRA reproduction and the submitted proposal scope.
- **W7-8 paper presentation is on the SAME RoLoRA paper** as the project — not a separate paper.
- **Compute** is DelftBlue + DAIC + local laptops. Plan from low- to high-cost. Llama-2-7B is a stretch goal only, not a baseline path.
- **Cluster access is available** on DelftBlue (and DAIC as backup). Open work is integration: real Slurm templates, module/account values, and the first paper-scale submission. See `docs/setup/delftblue.md` and `slurm/README.md` for the current state.
- **DelftBlue policy ground truths (as of 2026-05-25; do not re-litigate without evidence)**:
  - **Login nodes are restricted** — `uv sync`, `pip install`, light import sanity checks are OK; **NO heavy Python** (no `from_pretrained`, no training runs, no GPU access). For model / dataset downloads use `scripts/warm_caches.sh` which calls `huggingface_hub.snapshot_download` (file fetch only, no model load).
  - **Compute nodes have NO outbound internet for `huggingface.co`** (`[Errno 101] Network is unreachable`). All HF artefacts MUST be pre-cached on `/scratch/$USER/hf-cache` before submission. Training sbatch sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` defensively.
  - **`wandb.ai` IS reachable** from compute nodes — wandb live streaming works (verified). If it ever breaks, fallback is `WANDB_MODE=offline` + `wandb sync` from the login node after the job.
  - **`gpu-a100-small` partition caps**: `--mem-per-cpu` ≤ 8000 MB (NOT 8 GiB), `--cpus-per-task` ≤ 2, `--time` ≤ 4h (MIG-slice limit). All current C2 sbatch files match these caps. Going to full `gpu-a100` is the escalation if VRAM or wall time forces it.
  - **`eval.count_flops: False` in every reproduction YAML** — FederatedScope's default `True` invokes `fvcore.nn.FlopCountAnalysis` which pollutes the CUDA caching allocator and triggers `CUBLAS_STATUS_ALLOC_FAILED` on the 10 GB MIG slice. The FLOPS number is never reported anywhere in our paper artefacts.
- **Team is 3 people.** Split work by ownership layer (infrastructure / algorithm / improvement & analysis), not by experiment.

## Improvement directions the team committed to in the proposal
The proposal commits in writing to:
1. **Improved initialization** — orthogonal / SVD-based init for A.
2. **Separate learning rates for A and B** — LoRA+-style asymmetric LRs.
3. **Adaptive server-side optimization** — lightweight federated optimizer in place of plain averaging.

All three preserve RoLoRA's alternating structure. The deep-research plan's "partial participation + comm-time-aware scheduling" angle is **not** the chosen direction — keep it only as a contingency if the committed angles all fail to produce signal.

## Where context lives
- `docs/research/paper-rolora.pdf` — ground truth for paper claims.
- `docs/research/project-proposal.pdf` — what we promised the course (improvement angles, reproduction scope, timeline).
- `docs/research/lecture-01-introduction.pdf` — assessment split, deadlines, paper-selection rules.
- `docs/research/deep-research-plan.md` — compute budgets, week-by-week roadmap, kill criteria, code-availability assessment. Treat as authoritative for compute planning; treat its improvement-angle recommendation as **superseded** by the proposal.
- `docs/kickoff.md` — open questions for the team meeting.
- `docs/decisions/` — ADR log.

## Harness strategy
- **Primary** — the authors' OpenReview supplementary zip (`u4mobiHTJl`). Extracted (by the user) to `code/harness/rolora-supplement/` (gitignored — author code may not be redistributable).
- **Fallback** — our fork of `Pengxin-Guo/FedSA-LoRA` at `code/harness/fedsa-lora` (git submodule), used if the supplement is broken or fails the W2/W3 kill criteria (LoRA baseline off by >±2% on 3-client MNLI).
- Do **not** invest in FederatedScope-LLM. Do not clone `HuangOwen/RoLoRA` (different paper — EMNLP'24 quantization).

## Code conventions
- Python 3.10+, PyTorch 2.x, env managed by **uv** (`pyproject.toml` + `uv.lock`).
- `peft==0.10.0` is pinned (the deep-research plan flagged PEFT version drift as a real bug source).
- Config-driven experiments: YAML under `experiments/configs/`. Never flag-pollute training scripts.
- Assert exact-aggregation invariants during dev: `torch.equal(client_i.frozen_factor, server.frozen_factor)` at the top of every round. Tests in `tests/test_aggregation_invariants.py` enforce this on the helper functions.
- FFA-LoRA convention: B initialized to zero, A non-zero. Asserted in `tests/test_init_conventions.py`.

## Reproduction targets (per proposal)
- Table 1 (RoBERTa-Large, 5 GLUE tasks, multiple client counts).
- Figure 3 (50-client convergence curves).
- Incremental: start with one GLUE dataset (default MNLI), expand from there.
- Figure 2 MNIST toy is a laptop-CPU sanity check that must pass before any GPU is spent.

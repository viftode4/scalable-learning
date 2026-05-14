# OpenReview supplementary code — vendored source

The submitted proposal commits to using the authors' released code as the starting point. The official RoLoRA code is **not on public GitHub**; it was distributed as an OpenReview supplementary zip. We now vendor the extracted Apache-2.0 source under `code/harness/rolora-supplement/` so every teammate uses the same harness without a manual OpenReview download.

## Fresh clone setup
From the repo root:
```bash
make install-supplement
make supplement-smoke-all
```

`make install-supplement` creates an isolated Python 3.9 venv at `code/harness/rolora-supplement/RoLoRA-code/.venv-supplement`, applies `code/harness/rolora-supplement.patch` if needed, installs pinned LLM deps, and import-tests the result. `make supplement-smoke-all` runs the tiny RoBERTa-base smoke config in all three modes.

## Optional: refresh from OpenReview
1. Log into [openreview.net](https://openreview.net).
2. Navigate to the RoLoRA submission: `https://openreview.net/forum?id=u4mobiHTJl`.
3. Download `5662_Robust_Federated_Finetuni_Supplementary Material.zip` from the Code attachment.
4. Run `make supplement SUPPLEMENT_ZIP=/path/to/zip`, then re-run the audit and tests.

## Expected contents (confirmed 2026-05-14)
The audit (`docs/decisions/0004-supplement-audit.md`) found:
- **Zip filename** on OpenReview: `5662_Robust_Federated_Finetuni_Supplementary Material.zip`.
- **SHA256:** `ca9a64cb64bb48bb0a6dc35179760fe0e561dd566a474877ab42f65453aa4c11` — verify locally after download.
- 845 files, 16 MB, **Apache 2.0** license.
- Built on FederatedScope-LLM. Use `make supplement-smoke MODE=rolora` for a local RoBERTa-base smoke run.
- **Caveat:** Upstream RoLoRA alternation is hardcoded (no LoRA / FFA-LoRA baseline modes in-tree). Our tracked patch adds the three-mode switch used by the smoke config (see ADR 0004).

## Vendoring policy
`code/harness/rolora-supplement/` is tracked because the supplement is Apache-2.0 and teammate setup should be reproducible. The local `.venv-supplement/`, checkpoints, logs, caches, and benchmark data directories remain ignored.

## If the supplement is broken or unusable
The audit (ADR 0004) confirms the supplement runs and is the authors' real code. If FederatedScope-LLM proves too painful to wire up on DelftBlue (the deep-research plan flagged it as heavy), the contingency:
1. Use `code/harness/fedsa-lora/` (the submodule) for vanilla LoRA / FFA-LoRA baselines and port the supplement's `trainer.py` alternation into FedSA-LoRA for the RoLoRA arm.
2. Email the authors with the draft in `docs/templates/author-email.md` if a deeper bug needs clarification.
3. Record the harness pivot as a new ADR.

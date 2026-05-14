# OpenReview supplementary code — fetch & extract

The submitted proposal commits to using the authors' released code as the starting point. The official RoLoRA code is **not on public GitHub**; it lives only in the supplementary zip attached to the OpenReview submission.

## Action items
1. Log into [openreview.net](https://openreview.net).
2. Navigate to the RoLoRA submission: `https://openreview.net/forum?id=u4mobiHTJl`.
3. Scroll to the **Code** attachment under the Author-provided supplementary materials. Download the zip — it's named `5662_Robust_Federated_Finetuni_Supplementary Material.zip` (~16 MB).
4. From the repo root, run:
   ```bash
   make supplement
   make install-supplement
   make supplement-smoke-all
   ```
   `make supplement` auto-detects the real OpenReview filename in `~/Downloads` (or pass `SUPPLEMENT_ZIP=/path/to/zip`). `make install-supplement` creates an isolated Python 3.9 venv at `code/harness/rolora-supplement/RoLoRA-code/.venv-supplement`, applies `code/harness/rolora-supplement.patch` (adds the `SLS_ALTERNATION_MODE` switch), installs pinned LLM deps, and import-tests the result. `make supplement-smoke-all` runs the tiny RoBERTa-base smoke config in all three modes.

## Expected contents (confirmed 2026-05-14)
The audit (`docs/decisions/0004-supplement-audit.md`) found:
- **Zip filename** on OpenReview: `5662_Robust_Federated_Finetuni_Supplementary Material.zip`.
- **SHA256:** `ca9a64cb64bb48bb0a6dc35179760fe0e561dd566a474877ab42f65453aa4c11` — verify locally after download.
- 845 files, 16 MB, **Apache 2.0** license.
- Built on FederatedScope-LLM. Use `make supplement-smoke MODE=rolora` for a local RoBERTa-base smoke run.
- **Caveat:** Upstream RoLoRA alternation is hardcoded (no LoRA / FFA-LoRA baseline modes in-tree). Our tracked patch adds the three-mode switch used by the smoke config (see ADR 0004).

## Why this is gitignored
`code/harness/rolora-supplement/` is in `.gitignore`. Author code is Apache-2.0 per audit, but still kept out of git to avoid vendoring a large external tree. Each teammate fetches the supplement locally. The SHA256 hash of the zip is the only artifact we share — it lets us verify everyone is using the same source.

## If the supplement is broken or unusable
The audit (ADR 0004) confirms the supplement runs and is the authors' real code. If FederatedScope-LLM proves too painful to wire up on DelftBlue (the deep-research plan flagged it as heavy), the contingency:
1. Use `code/harness/fedsa-lora/` (the submodule) for vanilla LoRA / FFA-LoRA baselines and port the supplement's `trainer.py` alternation into FedSA-LoRA for the RoLoRA arm.
2. Email the authors with the draft in `docs/templates/author-email.md` if a deeper bug needs clarification.
3. Record the harness pivot as a new ADR.

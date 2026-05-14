# OpenReview supplementary code — fetch & extract

The submitted proposal commits to using the authors' released code as the starting point. The official RoLoRA code is **not on public GitHub**; it lives only in the supplementary zip attached to the OpenReview submission.

## Action items
1. Log into [openreview.net](https://openreview.net).
2. Navigate to the RoLoRA submission: `https://openreview.net/forum?id=u4mobiHTJl`.
3. Scroll to the **Code** attachment under the Author-provided supplementary materials. Download the zip.
4. Drop the zip at `~/Downloads/rolora-supplement.zip` (or pass an explicit path to the extractor in step 5).
5. From the repo root:
   ```bash
   bash scripts/extract_supplement.sh
   # or, with a custom path:
   bash scripts/extract_supplement.sh /path/to/zip
   ```
   The script unpacks into `code/harness/rolora-supplement/`, prints SHA256 + a content sanity check.

## Expected contents (confirmed 2026-05-14)
The audit (`docs/decisions/0004-supplement-audit.md`) found:
- **Zip filename** on OpenReview: `5662_Robust_Federated_Finetuni_Supplementary Material.zip`.
- **SHA256:** `ca9a64cb64bb48bb0a6dc35179760fe0e561dd566a474877ab42f65453aa4c11` — verify locally after download.
- 845 files, 16 MB, **Apache 2.0** license.
- Built on FederatedScope-LLM. Example run:
  ```bash
  cd code/harness/rolora-supplement/RoLoRA-code
  cd sst2 && python qnli2json.py && cd ..
  python federatedscope/main.py --cfg federatedscope/llm/baseline/test_glue.yaml
  ```
- **Caveat:** RoLoRA alternation is hardcoded (no LoRA / FFA-LoRA baseline modes in-tree). Plan is to patch `federatedscope/llm/trainer/trainer.py` to add config-driven baseline selection (see ADR 0004).

## Why this is gitignored
`code/harness/rolora-supplement/` is in `.gitignore`. Author code may not be redistributable (license unclear from the OpenReview page). Each teammate fetches the supplement locally. The SHA256 hash of the zip is the only artifact we share — it lets us verify everyone is using the same source.

## If the supplement is broken or unusable
The audit (ADR 0004) confirms the supplement runs and is the authors' real code. If FederatedScope-LLM proves too painful to wire up on DelftBlue (the deep-research plan flagged it as heavy), the contingency:
1. Use `code/harness/fedsa-lora/` (the submodule) for vanilla LoRA / FFA-LoRA baselines and port the supplement's `trainer.py` alternation into FedSA-LoRA for the RoLoRA arm.
2. Email the authors with the draft in `docs/templates/author-email.md` if a deeper bug needs clarification.
3. Record the harness pivot as a new ADR.

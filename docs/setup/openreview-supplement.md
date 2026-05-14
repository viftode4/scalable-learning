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

## Expected contents
The extractor warns and exits non-zero if no Python files are found. Typical RoLoRA supplements include:
- A training entrypoint (FederatedScope-LLM style or a custom federated loop).
- LoRA / FFA-LoRA / RoLoRA implementations.
- Configs / shell scripts that reproduce paper tables.

## Why this is gitignored
`code/harness/rolora-supplement/` is in `.gitignore`. Author code may not be redistributable (license unclear from the OpenReview page). Each teammate fetches the supplement locally. The SHA256 hash of the zip is the only artifact we share — it lets us verify everyone is using the same source.

## If the supplement is broken or unusable
This is plausible — the deep-research plan flags RoLoRA's code as likely a "research-grade dump rather than a polished release." The contingency:
1. Try the FedSA-LoRA submodule at `code/harness/fedsa-lora/` — it implements LoRA / FFA-LoRA / FedSA-LoRA on RoBERTa+GLUE and is well-suited as a baseline harness.
2. Email the authors (use the draft in `docs/templates/author-email.md`); the deep-research plan points at `akhisti@ece.utoronto.ca`.
3. Record the pivot in `docs/decisions/` (new ADR).

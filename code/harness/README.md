# Harness checkouts

Two harnesses, used in priority order. See `docs/decisions/0001-harness-strategy.md` for the rationale.

## `rolora-supplement/` — primary (authors' code)
The OpenReview supplementary zip for RoLoRA (forum `u4mobiHTJl`). Required by the project proposal ("we will use the authors' released code as the starting point").

**Not in git** — see `.gitignore`. Author code may not be redistributable; we track it locally only.

To get it on a fresh clone, follow `docs/setup/openreview-supplement.md`. After extraction:
```bash
find code/harness/rolora-supplement -name '*.py' | head
```
should list Python files. If empty, the supplement download failed — re-check the OpenReview steps.

## `fedsa-lora/` — backup harness (git submodule)
Git submodule pointing at our fork `viftode4/FedSA-LoRA` (upstream: `Pengxin-Guo/FedSA-LoRA`, ICLR 2025).

Why it's here: FedSA-LoRA already implements LoRA, FFA-LoRA, and FedSA-LoRA on RoBERTa-base/large for GLUE under FedAvg with Dirichlet-α non-IID splits. The deep-research plan flags that the OpenReview supplement may be a research-grade dump; FedSA-LoRA is a vetted public alternative that the W2 kill criterion explicitly authorizes us to pivot to.

```bash
# Initialize after clone
git submodule update --init --recursive

# Pull upstream changes from Pengxin-Guo/FedSA-LoRA into our fork:
cd code/harness/fedsa-lora
git remote add upstream https://github.com/Pengxin-Guo/FedSA-LoRA.git  # one-time
git fetch upstream
git merge upstream/main          # or rebase
git push                          # pushes to viftode4/FedSA-LoRA
cd ../../..
git add code/harness/fedsa-lora && git commit  # records the new submodule SHA
```

When we start modifying the submodule, create a branch in `viftode4/FedSA-LoRA` named `sls-rolora/main` and point the submodule at it.

## Do not clone
- `HuangOwen/RoLoRA` — different paper (EMNLP'24, quantization).
- `alibaba/FederatedScope/tree/llm` — heavy, older codebase; the deep-research plan rejects it as the main vehicle.

# Harness — FedSA-LoRA fork (stub)

This directory will hold a **git submodule** pointing to our fork of `Pengxin-Guo/FedSA-LoRA`. The fork has not been created yet — the actual `gh repo fork` and `git submodule add` is deferred until the team confirms in or after the kickoff meeting.

## Why FedSA-LoRA
The deep-research plan (see `docs/research/deep-research-plan.md`) selected this as the harness because it:
- already implements LoRA, FFA-LoRA, and FedSA-LoRA baselines on RoBERTa-base/large for GLUE,
- supports FedAvg with Dirichlet-α non-IID client splits out of the box,
- is small, public (ICLR 2025), Python 3.10 + PyTorch 2.1,
- requires roughly 50 lines of code to add RoLoRA's odd/even alternation.

We will cross-check against `CERT-Lab/fed-sb` (which contains a third-party RoLoRA implementation) for numerical sanity.

## Intended commands (do not run yet)
```bash
# 1. Fork the upstream repo on GitHub
gh repo fork Pengxin-Guo/FedSA-LoRA --clone=false --org=viftode4

# 2. Add as submodule under this directory
git submodule add git@github.com:viftode4/FedSA-LoRA.git code/harness/fedsa-lora
git submodule update --init --recursive
```

After execution, capture the decision (submodule vs. vendor, fork ownership, branch strategy) in `docs/decisions/`.

## Do NOT
- Clone `HuangOwen/RoLoRA` — different paper (EMNLP'24, quantization).
- Adopt FederatedScope-LLM as the main vehicle — heavy dependency footprint, older than FedSA-LoRA, would burn week 1.

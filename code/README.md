# Code

This directory will hold the federated-LoRA training code: a fork of FedSA-LoRA (see `harness/`) plus our additions implementing RoLoRA's odd/even alternation and the chosen improvement angle.

## Planned layout (after kickoff)
```
code/
  harness/               # Fork of Pengxin-Guo/FedSA-LoRA (git submodule)
  src/
    rolora/              # Our alternation logic on top of the harness
    improvements/        # Partial-participation + comm-time-aware scheduling
    eval/                # GLUE eval + plotting helpers
  tests/                 # Unit tests + exactness asserts (torch.equal of frozen factors across clients)
```

## Env (planned)
- Python 3.10+
- PyTorch 2.x with CUDA matching DelftBlue's `gpu-a100-small` partition
- `transformers`, `peft`, `datasets`, `accelerate`
- Pinned via `requirements.txt` or `pyproject.toml` once the harness is added.

## Not in scope here
- Slurm job scripts live in `slurm/`.
- Experiment configs live in `experiments/configs/`.
- Outputs land in `results/` (gitignored).

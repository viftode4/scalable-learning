# Slurm templates — DelftBlue + DAIC

Templates for the two partitions we expect to use most.

| File | Partition | When |
|---|---|---|
| `gpu-a100-small.sbatch` | DelftBlue `gpu-a100-small` (10 GB MIG slice, ≤4h) | Default for RoBERTa-Large LoRA — fastest queue |
| `gpu-v100.sbatch` | DelftBlue `gpu-v100` (32 GB) | Backup when A100-small queue saturated or VRAM tight |

## Status — UNCONFIRMED
Partition names and `module load` lines are placeholders. They are based on the deep-research plan, **not** on TA instructions (which haven't arrived). When Dennis Heijmans / Rui Wang send the access flow, update both `.sbatch` files and remove this banner.

## Usage
```bash
sbatch slurm/gpu-a100-small.sbatch experiments/configs/template_roberta_mnli.yaml
```

`slurm-<jobid>.out` files are gitignored.

## Do not use yet
- `gpu-a100` (full 80 GB) — longest queues; reserved for Llama-7B stretch only.
- DAIC — backup capacity; we'll add a template if we actually hit DelftBlue limits.

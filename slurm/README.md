# Slurm scripts — DelftBlue + DAIC

Job scripts for the TU Delft DelftBlue and DAIC clusters.

## DelftBlue partitions we'll use
- `gpu-a100-small` — 10 GB MIG slice, ≤4h jobs, ≤1 GPU. **Default for RoBERTa-Large LoRA** (≤6 GB at fp16/bs=16). Fastest queue.
- `gpu-v100` — 32 GB V100s. Backup if A100 small queue is saturated.
- `gpu-a100` — full 80 GB A100. Reserved for the Llama-2-7B stretch target only.

## DAIC
Backup capacity. INSY useful for 4h iteration jobs.

## Get a faculty share early
Default low-priority jobs may queue 8–24h on busy days. Request a faculty share via TOPdesk in week 1 — see `docs/kickoff.md` for the action item.

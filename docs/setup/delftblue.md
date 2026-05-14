# DelftBlue / DAIC access

## Status: waiting for TAs

The CS 4725 lecture-1 slides say:
> Request access to DelftBlue or DAIC — **The TAs will provide the instructions.**

Until **Dennis Heijmans** or **Rui Wang** share the access flow in a week-4 lecture or a BrightSpace announcement, **do not file a TOPdesk request blindly**. Track BrightSpace + course email; if nothing arrives by mid-W4, ping Dennis directly.

When instructions arrive, paste them here verbatim and replace this banner.

---

## UNCONFIRMED — provisional partition info

The text below is from the deep-research plan, useful for context but **not authoritative** until TAs confirm. Do not change Slurm template partition names based on this alone.

### DelftBlue partitions we expect to use
| Partition | VRAM | Time limit | When |
|---|---|---|---|
| `gpu-a100-small` | 10 GB MIG slice | ≤ 4h | Default for RoBERTa-Large LoRA (fits in ≤6 GB at fp16/bs=16). Fastest queue. |
| `gpu-v100` | 32 GB | longer | Backup when `gpu-a100-small` saturated, or higher rank / longer sequence. |
| `gpu-a100` | 80 GB | longer | Reserved for Llama-7B stretch only — longest queues, do not use otherwise. |

### Operational notes (provisional)
- Default jobs run at low priority — 8–24h queue on busy days. A faculty share request (TOPdesk) would mitigate; **wait for TA guidance before filing one**.
- Cache tokenized datasets on `/scratch/$USER/sls-data` (the Slurm templates do this).
- `slurm-<jobid>.out` files are gitignored. Inspect locally; archive interesting ones into `results/`.

### DAIC
Backup capacity if DelftBlue queues clog. Access flow is also TA-driven.

## When instructions arrive, fill these in
- Account/group on DelftBlue: TBD
- `module load` lines for the Python/CUDA stack we're allowed: TBD
- SSH config snippet: TBD
- Faculty-share request flow (if any): TBD
- DAIC: TBD

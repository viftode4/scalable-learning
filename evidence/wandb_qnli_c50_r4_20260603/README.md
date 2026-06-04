# W&B qnli_c50_r4 audit — 2026-06-03

- Runs visible: 28
- Series points exported: 560
- All visible runs are group `qnli_c50_r4`.
- Local matching Slurm stdout/stderr was not found in this checkout; W&B is the only evidence for these 28 runs.

## Aggregate by method / LR

| method | lr | n | seeds | states | target rounds | final/best accs | mean best | max best |
|---|---:|---:|---|---|---|---|---:|---:|
| ffa_lora | 1e-2 | 3 | 0,1,2 | crashed | 20 | s0 final 0.5222 / best 0.5467, s1 final 0.5095 / best 0.5703, s2 final 0.5267 / best 0.5295 | 0.5488 | 0.5703 |
| ffa_lora | 2e-2 | 3 | 0,1,2 | crashed | 20 | s0 final 0.5458 / best 0.5639, s1 final 0.4805 / best 0.5657, s2 final 0.5231 / best 0.5394 | 0.5564 | 0.5657 |
| ffa_lora | 5e-3 | 1 | 0 | crashed | 20 | s0 final 0.5050 / best 0.5331 | 0.5331 | 0.5331 |
| lora | 1e-2 | 3 | 0,1,2 | crashed | 20 | s0 final 0.5231 / best 0.5449, s1 final 0.5068 / best 0.5712, s2 final 0.5267 / best 0.5304 | 0.5488 | 0.5712 |
| lora | 2e-2 | 3 | 0,1,2 | crashed | 20 | s0 final 0.5440 / best 0.5703, s1 final 0.4814 / best 0.5712, s2 final 0.5231 / best 0.5358 | 0.5591 | 0.5712 |
| lora | 5e-3 | 1 | 0 | crashed | 20 | s0 final 0.5032 / best 0.5340 | 0.5340 | 0.5340 |
| rolora | 1e-1 | 1 | 0 | crashed | 20 | s0 final 0.4796 / best 0.5258 | 0.5258 | 0.5258 |
| rolora | 1e-2 | 3 | 0,1,2 | crashed | 20 | s0 final 0.8214 / best 0.8214, s1 final 0.8713 / best 0.8713, s2 final 0.8604 / best 0.8604 | 0.8510 | 0.8713 |
| rolora | 1e-3 | 1 | 0 | crashed | 20 | s0 final 0.5195 / best 0.5195 | 0.5195 | 0.5195 |
| rolora | 2e-2 | 3 | 0,1,2 | crashed | 20 | s0 final 0.8586 / best 0.8640, s1 final 0.7162 / best 0.7715, s2 final 0.8613 / best 0.8613 | 0.8323 | 0.8640 |
| rolora | 2e-3 | 1 | 0 | crashed | 20 | s0 final 0.5612 / best 0.5612 | 0.5612 | 0.5612 |
| rolora | 5e-2 | 1 | 0 | crashed | 20 | s0 final 0.6355 / best 0.7715 | 0.7715 | 0.7715 |
| rolora | 5e-3 | 3 | 0,1,2 | crashed | 20 | s0 final 0.7416 / best 0.8486, s1 final 0.8432 / best 0.8441, s2 final 0.8595 / best 0.8595 | 0.8507 | 0.8595 |
| rolora | 5e-4 | 1 | 0 | crashed | 20 | s0 final 0.4787 / best 0.4832 | 0.4832 | 0.4832 |

## Best runs overall

- 0.8713 best / 0.8713 final — `rolora_lr1e-2_seed1` state=crashed target_rounds=20 final_round=19 runtime_h=3.85
- 0.8640 best / 0.8586 final — `rolora_lr2e-2_seed0` state=crashed target_rounds=20 final_round=19 runtime_h=3.86
- 0.8613 best / 0.8613 final — `rolora_lr2e-2_seed2` state=crashed target_rounds=20 final_round=19 runtime_h=3.87
- 0.8604 best / 0.8604 final — `rolora_lr1e-2_seed2` state=crashed target_rounds=20 final_round=19 runtime_h=3.88
- 0.8595 best / 0.8595 final — `rolora_lr5e-3_seed2` state=crashed target_rounds=20 final_round=19 runtime_h=3.86
- 0.8486 best / 0.7416 final — `rolora_lr5e-3_seed0` state=crashed target_rounds=20 final_round=19 runtime_h=3.86
- 0.8441 best / 0.8432 final — `rolora_lr5e-3_seed1` state=crashed target_rounds=20 final_round=19 runtime_h=3.90
- 0.8214 best / 0.8214 final — `rolora_lr1e-2_seed0` state=crashed target_rounds=20 final_round=19 runtime_h=3.87
- 0.7715 best / 0.6355 final — `rolora_lr5e-2_seed0` state=crashed target_rounds=20 final_round=19 runtime_h=3.86
- 0.7715 best / 0.7162 final — `rolora_lr2e-2_seed1` state=crashed target_rounds=20 final_round=19 runtime_h=3.87
- 0.5712 best / 0.5068 final — `lora_lr1e-2_seed1` state=crashed target_rounds=20 final_round=19 runtime_h=3.88
- 0.5712 best / 0.4814 final — `lora_lr2e-2_seed1` state=crashed target_rounds=20 final_round=19 runtime_h=3.91

## Paper Table 1 comparison for QNLI / rank 4 / 50 clients

Paper target numbers from `docs/research/paper-rolora.pdf`, Table 1:

| method | paper QNLI acc | best W&B acc here | mean best over available seeds/LRs | assessment |
|---|---:|---:|---:|---|
| LoRA | 78.13 ± 5.13 | 57.12 | 55.91 at best LR (`2e-2`) | Not reproduced; far below paper baseline. |
| FFA-LoRA | 85.05 ± 0.34 | 57.03 | 55.64 at best LR (`2e-2`) | Not reproduced; far below paper baseline. |
| RoLoRA | 90.00 ± 0.63 | 87.13 | 85.10 at `1e-2`, 85.07 at `5e-3` | Strong completed 20-round RoLoRA baseline; below paper Table 1 and shorter than the repo's 30-round C50 plan. |

Caveats:
- These W&B runs have `total_round_num=20`; histories contain rounds 0–19, so the configured 20-round trajectories completed. The current repo paper-aligned config for 50-client cells says 30 rounds, so this is a completed 20-round baseline, not the full planned 30-round C50 cell.
- All visible runs are marked `crashed`, with runtime near the 4h `gpu-a100-small` cap. Given the complete 0–19 histories, treat this as Slurm walltime termination during/after finalization unless recovered cluster logs show otherwise.
- The run config exposed by W&B only records `alternation_mode`, `client_num`, `seed`, and `total_round_num`; optimizer/LR appears in the run name/tags but not in full config, so cluster command/log recovery is still needed for audit quality.
- No matching Slurm stdout/stderr logs are present in this local checkout.

## Branch provenance correction

After inspecting `origin/fix-rolora`, these W&B runs most likely came from Daniel's branch, not current `main`:

- `origin/fix-rolora:experiments/configs/repro_qnli_c50_r4_lr1e-2.yaml` uses `model.type: 'roberta-base@huggingface_llm'`.
- The branch Slurm files set W&B names like `rolora_lr1e-2_seed${SEED}` under group `qnli_c50_r4`, matching the W&B run names.
- Therefore these runs should be treated as **QNLI / 50 clients / rank 4 / RoBERTa-base / 20 rounds**, unless recovered Slurm logs prove otherwise.

Consequence: they are useful as fast reproduction/improvement evidence and a control baseline, but they are **not direct RoBERTa-Large Table 1 reproduction evidence**.

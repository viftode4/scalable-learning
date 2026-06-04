# WhatsApp plot bundle — 2026-06-04

Scope: QNLI / RoBERTa-base / 50 clients / rank 4 / 20 rounds proxy.
Not paper-scale RoBERTa-Large Table 1.

- Old RoLoRA lr=1e-2 seed0 final: 82.14%.
- Old RoLoRA lr=1e-2 3-seed mean final: 85.10% ± 2.97 CI95.
- Orth-A + default AB seed0 final: 82.94%.
- Orth-A + BBA seed0 final: 88.54%.
- Gain vs Orth-A+AB seed0: +5.60 pp.
- Gain vs old RoLoRA seed0: +6.40 pp.
- BBA+Orth seed1 currently parsed through round 15; latest test 87.61%.

Files:
- `01_main_curve_old_vs_improvements.png`
- `02_final_accuracy_bars.png`
- `03_phase_mechanics_bba.png`

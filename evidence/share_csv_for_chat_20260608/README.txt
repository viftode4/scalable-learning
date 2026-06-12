CSV bundle for chat/debugging RoLoRA logging issue.

Most important files to attach first:
1. improvement_server_curves/*__server_metrics.csv
   - True 20-round curves for improvement runs. Use `round`, not W&B `_step`.
2. baseline_wandb_export/runs_summary.csv
   - Baseline run inventory and final/best metrics from W&B.
3. baseline_wandb_export/server_history.csv
   - Baseline per-round W&B server histories.

Optional diagnostics:
- improvement_metadata/*__run_metadata.csv: config/run provenance.
- improvement_phase_aggregation/*__phase_markers.csv: phase per client/round.
- improvement_phase_aggregation/*__aggregation_monitor.csv: aggregation monitor rows.

Ignored/bad run: proxy_phase_bba_orth_a_c50_r4_lr1e-2_seed2 (5 seconds, no history) is intentionally not included.

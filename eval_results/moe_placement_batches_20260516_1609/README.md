# MoE Placement Eval Batches - 2026-05-16

This directory groups the MoE placement experiment outputs that were previously
spread directly under `eval_results/`.

## Layout

- `early_full_max8_ws4/`: early-morning `max_tokens=8`, `max_num_batched_tokens=512`, `max_num_seqs=1` full-layer attempts and the completed `full_ep4_fiws_20260516_0230` ws=4 run.
- `early_dry_checks/`: dry-run world-size and scale checks from the same early-morning setup.
- `early_perf_ws8_ws4_0228/`: completed early-morning ws=8/ws=4 perf run with `max_tokens=32`, including routing profiles, placement JSONs, and analysis.
- `restats_early_morning/`: regenerated CSV/Markdown summaries for the early-morning results.
- `afternoon_retry_ws4_round_robin/`: retry/debug run for the ws=4 EP-HT round-robin failure.
- `afternoon_dry_rerun_checks/`: dry-run rerun/port-check scale sweep.
- `afternoon_perf_ws8_ws4_rerun_1428/`: afternoon ws=8/ws=4 rerun batch starting at 14:28.
- `afternoon_perf_ws8_ws4_rerun_1516/`: afternoon rerun batch starting at 15:16, including top-level `aime24_ep_*` outputs that were overwritten by this latest batch.
- `smoke_moe_pr_20260514/`: earlier smoke/PR validation outputs from 2026-05-14.

## Note

Some `summary.json` and log files preserve the paths that existed at runtime,
such as `eval_results/moe_routing_profile_*.json`. The files themselves have
been moved into the grouped batch directories above.

# POMO Performance Audit Report

- Config: `D:\Uni Material\Code_Odyssey\Code\DQN\260407\configs\default_config.yaml`
- Baseline config: `D:\Uni Material\Code_Odyssey\Code\DQN\260330\configs\default_config.yaml`
- Timestamp: `2026-04-10T22:17:56`

## Main Findings

- Effective rollout size grows as `batch_size * num_pomo_starts * (8 if augment_8fold else 1)`.
- With current settings (`num_pomo_starts=10`, `augment_8fold=True`), a safe-equivalent batch from reference expanded batch `1280` is approximately `16`.
- Real-data mixed sampling builds `distance_km` and `travel_time` via Python loops and temporary float64 NumPy arrays before conversion, which increases CPU time and host memory churn.
- Loading many mixed depot/day problems up-front also increases startup RAM compared to synthetic-only generation.

## Baseline Comparison

- Old `num_pomo_starts`: `20`
- Old `augment_8fold`: `False`
- New `num_pomo_starts`: `10`
- New `augment_8fold`: `True`
- Relative effective multiplier (new vs old, same batch size): `4.00x`

## Dry-Run Measurements

| Scenario | Batch | Expanded Batch | Gen(s) | Rollout(s) | Instance MB | Rollout MB | Estimated Matrix MB | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| new_current | 4 | 320 | 0.054 | 0.461 | 0.04 | 0.03 | 6.35 | none |
| new_current | 8 | 640 | 0.002 | 0.583 | 0.08 | 0.06 | 12.70 | none |
| new_no_augmentation | 4 | 40 | 0.001 | 0.089 | 0.00 | 0.00 | 0.79 | none |
| new_no_augmentation | 8 | 80 | 0.001 | 0.101 | 0.01 | 0.01 | 1.59 | none |
| synthetic_only | 4 | 40 | 0.014 | 0.198 | 0.09 | 0.00 | 0.79 | none |
| synthetic_only | 8 | 80 | 0.001 | 0.368 | 0.17 | 0.01 | 1.59 | none |

## Recommendations

- For stability, start with `--batch-size 8` or `16` when `augment_8fold=true` and `num_pomo_starts=10`.
- If you need larger batch size, disable augmentation during initial training and re-enable later.
- Cache or pre-materialize per-instance submatrices once per sampled customer set if you keep real-matrix training.
- Replace per-batch Python loops with vectorized sampling/indexing where possible.
- Keep this audit script in CI/experiments to detect future regressions quickly.

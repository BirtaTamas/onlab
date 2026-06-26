# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `5`
- rows: `168`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.398251 | 0.192909 | 0.548475 | 0.636905 | 0.601749 |
| xgboost | 0.344685 | 0.150727 | 0.455555 | 0.779762 | 0.655315 |

## Closer Per Tick

- lstm: `0`
- xgboost: `168`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

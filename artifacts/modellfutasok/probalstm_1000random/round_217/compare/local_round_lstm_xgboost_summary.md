# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `11`
- rows: `164`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.072128 | 0.028969 | 0.093780 | 0.993902 | 0.072128 |
| xgboost | 0.112826 | 0.036575 | 0.140064 | 0.908537 | 0.112826 |

## Closer Per Tick

- lstm: `164`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

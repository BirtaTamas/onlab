# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `9`
- rows: `270`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.166516 | 0.057321 | 0.205441 | 1.000000 | 0.166516 |
| xgboost | 0.248569 | 0.097712 | 0.318130 | 0.981481 | 0.248569 |

## Closer Per Tick

- lstm: `256`
- xgboost: `14`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

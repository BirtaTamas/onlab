# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `2`
- rows: `169`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.031248 | 0.005585 | 0.034561 | 1.000000 | 0.031248 |
| xgboost | 0.083324 | 0.015295 | 0.092581 | 1.000000 | 0.083324 |

## Closer Per Tick

- lstm: `168`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

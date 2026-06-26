# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `11`
- rows: `151`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.494125 | 0.284268 | 0.757249 | 0.476821 | 0.494125 |
| xgboost | 0.521347 | 0.294651 | 0.781182 | 0.423841 | 0.521347 |

## Closer Per Tick

- lstm: `85`
- xgboost: `66`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `9`
- rows: `252`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.811620 | 0.726532 | 2.837315 | 0.134921 | 0.188380 |
| xgboost | 0.711395 | 0.573164 | 1.631557 | 0.150794 | 0.288605 |

## Closer Per Tick

- lstm: `2`
- xgboost: `250`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

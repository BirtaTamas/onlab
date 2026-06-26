# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `4`
- rows: `106`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.348071 | 0.149913 | 0.460239 | 0.726415 | 0.348071 |
| xgboost | 0.325744 | 0.137202 | 0.427213 | 0.716981 | 0.325744 |

## Closer Per Tick

- lstm: `37`
- xgboost: `69`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

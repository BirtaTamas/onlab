# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `2`
- rows: `205`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.035492 | 0.006919 | 0.039607 | 1.000000 | 0.035492 |
| xgboost | 0.062222 | 0.009178 | 0.067540 | 1.000000 | 0.062222 |

## Closer Per Tick

- lstm: `192`
- xgboost: `13`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `5`
- rows: `238`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.171809 | 0.089618 | 0.244477 | 0.768908 | 0.171809 |
| xgboost | 0.168044 | 0.086127 | 0.241097 | 0.785714 | 0.168044 |

## Closer Per Tick

- lstm: `187`
- xgboost: `51`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

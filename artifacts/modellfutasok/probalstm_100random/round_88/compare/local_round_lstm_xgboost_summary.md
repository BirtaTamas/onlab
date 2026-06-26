# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-Z9VnvF_JkEDX6y_HyMsFXx/aurora-vs-heroic-m3-mirage.csv`
- round_num: `10`
- rows: `158`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.046221 | 0.002269 | 0.047396 | 1.000000 | 0.953779 |
| xgboost | 0.018568 | 0.000374 | 0.018758 | 1.000000 | 0.981432 |

## Closer Per Tick

- lstm: `0`
- xgboost: `158`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

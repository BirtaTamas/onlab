# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `2`
- rows: `204`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.047324 | 0.002374 | 0.048556 | 1.000000 | 0.952676 |
| xgboost | 0.020293 | 0.000420 | 0.020506 | 1.000000 | 0.979707 |

## Closer Per Tick

- lstm: `0`
- xgboost: `204`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

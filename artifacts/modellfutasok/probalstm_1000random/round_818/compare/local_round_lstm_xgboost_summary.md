# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `7`
- rows: `146`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.330693 | 0.158897 | 0.454236 | 0.691781 | 0.669307 |
| xgboost | 0.297788 | 0.131256 | 0.394506 | 1.000000 | 0.702212 |

## Closer Per Tick

- lstm: `7`
- xgboost: `139`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

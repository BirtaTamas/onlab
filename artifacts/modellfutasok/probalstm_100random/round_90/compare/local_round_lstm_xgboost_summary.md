# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `11`
- rows: `209`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.379792 | 0.159543 | 0.494000 | 0.933014 | 0.620208 |
| xgboost | 0.358934 | 0.143047 | 0.458885 | 0.990431 | 0.641066 |

## Closer Per Tick

- lstm: `39`
- xgboost: `170`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-mouz-bo3-D4mE8XcULbH9iT3IhMhdJY/legacy-vs-mouz-m1-ancient.csv`
- round_num: `6`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.137978 | 0.046907 | 0.171582 | 1.000000 | 0.862022 |
| xgboost | 0.112206 | 0.043318 | 0.143488 | 1.000000 | 0.887794 |

## Closer Per Tick

- lstm: `27`
- xgboost: `203`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `10`
- rows: `135`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.512022 | 0.294200 | 0.777882 | 0.385185 | 0.487978 |
| xgboost | 0.367905 | 0.162745 | 0.490887 | 0.585185 | 0.632095 |

## Closer Per Tick

- lstm: `0`
- xgboost: `135`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

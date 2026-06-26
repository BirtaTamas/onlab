# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `2`
- rows: `176`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.585063 | 0.386584 | 1.021893 | 0.323864 | 0.414937 |
| xgboost | 0.510475 | 0.299072 | 0.792520 | 0.301136 | 0.489525 |

## Closer Per Tick

- lstm: `16`
- xgboost: `160`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

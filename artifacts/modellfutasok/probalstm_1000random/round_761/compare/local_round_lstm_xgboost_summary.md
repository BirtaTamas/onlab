# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `9`
- rows: `197`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.458392 | 0.258192 | 0.682120 | 0.233503 | 0.458392 |
| xgboost | 0.433986 | 0.234252 | 0.635710 | 0.563452 | 0.433986 |

## Closer Per Tick

- lstm: `62`
- xgboost: `135`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

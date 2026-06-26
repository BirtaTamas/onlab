# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `6`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.515407 | 0.301283 | 0.790545 | 0.265217 | 0.484593 |
| xgboost | 0.395479 | 0.188542 | 0.541115 | 0.413043 | 0.604521 |

## Closer Per Tick

- lstm: `2`
- xgboost: `228`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

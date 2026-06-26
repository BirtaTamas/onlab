# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `11`
- rows: `162`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.482870 | 0.240812 | 0.671456 | 0.654321 | 0.517130 |
| xgboost | 0.379464 | 0.149311 | 0.483089 | 1.000000 | 0.620536 |

## Closer Per Tick

- lstm: `0`
- xgboost: `162`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

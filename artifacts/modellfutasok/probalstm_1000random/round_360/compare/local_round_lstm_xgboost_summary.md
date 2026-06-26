# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `5`
- rows: `247`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.330021 | 0.185111 | 0.488408 | 0.453441 | 0.330021 |
| xgboost | 0.321029 | 0.169562 | 0.461700 | 0.635628 | 0.321029 |

## Closer Per Tick

- lstm: `128`
- xgboost: `119`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

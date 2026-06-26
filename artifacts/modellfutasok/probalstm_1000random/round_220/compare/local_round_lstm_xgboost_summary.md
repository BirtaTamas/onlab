# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `14`
- rows: `163`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.141631 | 0.022968 | 0.154667 | 1.000000 | 0.858369 |
| xgboost | 0.019451 | 0.000401 | 0.019654 | 1.000000 | 0.980549 |

## Closer Per Tick

- lstm: `0`
- xgboost: `163`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

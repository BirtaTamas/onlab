# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `3`
- rows: `275`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.195700 | 0.071429 | 0.245641 | 0.923636 | 0.195700 |
| xgboost | 0.178954 | 0.056668 | 0.217034 | 0.963636 | 0.178954 |

## Closer Per Tick

- lstm: `137`
- xgboost: `138`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

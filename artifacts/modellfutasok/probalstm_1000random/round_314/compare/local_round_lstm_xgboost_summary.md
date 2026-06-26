# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `21`
- rows: `233`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.516279 | 0.295459 | 0.810515 | 0.480687 | 0.483721 |
| xgboost | 0.397641 | 0.180019 | 0.545490 | 0.884120 | 0.602359 |

## Closer Per Tick

- lstm: `15`
- xgboost: `218`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

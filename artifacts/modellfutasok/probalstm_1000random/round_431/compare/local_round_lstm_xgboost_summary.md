# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `18`
- rows: `146`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.016634 | 0.000674 | 0.016984 | 1.000000 | 0.016634 |
| xgboost | 0.070026 | 0.009824 | 0.075495 | 1.000000 | 0.070026 |

## Closer Per Tick

- lstm: `105`
- xgboost: `41`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `16`
- rows: `252`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.598422 | 0.376077 | 0.957267 | 0.115079 | 0.598422 |
| xgboost | 0.645958 | 0.445906 | 1.132358 | 0.154762 | 0.645958 |

## Closer Per Tick

- lstm: `189`
- xgboost: `63`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

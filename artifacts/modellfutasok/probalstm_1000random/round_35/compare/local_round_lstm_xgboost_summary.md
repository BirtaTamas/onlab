# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `4`
- rows: `220`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.269644 | 0.131743 | 0.370829 | 0.740909 | 0.269644 |
| xgboost | 0.280579 | 0.139943 | 0.389781 | 0.568182 | 0.280579 |

## Closer Per Tick

- lstm: `171`
- xgboost: `49`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

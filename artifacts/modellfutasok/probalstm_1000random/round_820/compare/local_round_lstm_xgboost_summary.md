# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `6`
- rows: `174`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.664944 | 0.482676 | 1.234432 | 0.143678 | 0.335056 |
| xgboost | 0.612630 | 0.411222 | 1.038347 | 0.270115 | 0.387370 |

## Closer Per Tick

- lstm: `50`
- xgboost: `124`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

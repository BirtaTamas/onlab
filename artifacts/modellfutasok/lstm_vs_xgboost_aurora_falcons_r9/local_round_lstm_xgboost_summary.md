# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\esports_world_cup\esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW\aurora-vs-falcons-m2-train.csv`
- round_num: `9`
- rows: `184`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.343572 | 0.182945 | 0.515250 | 0.679348 | 0.656428 |
| xgboost | 0.248344 | 0.094108 | 0.315167 | 0.994565 | 0.751656 |

## Closer Per Tick

- lstm: `5`
- xgboost: `179`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

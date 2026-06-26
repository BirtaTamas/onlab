# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `14`
- rows: `123`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.059767 | 0.004070 | 0.061910 | 1.000000 | 0.940233 |
| xgboost | 0.014756 | 0.000258 | 0.014887 | 1.000000 | 0.985244 |

## Closer Per Tick

- lstm: `0`
- xgboost: `123`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

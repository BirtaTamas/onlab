# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `8`
- rows: `270`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.069627 | 0.013696 | 0.078208 | 1.000000 | 0.069627 |
| xgboost | 0.064643 | 0.007923 | 0.069088 | 1.000000 | 0.064643 |

## Closer Per Tick

- lstm: `186`
- xgboost: `84`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

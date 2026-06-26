# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `2`
- rows: `246`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.154064 | 0.042707 | 0.180908 | 1.000000 | 0.154064 |
| xgboost | 0.210648 | 0.072931 | 0.259146 | 1.000000 | 0.210648 |

## Closer Per Tick

- lstm: `232`
- xgboost: `14`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m1-inferno-p4.csv`
- round_num: `5`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.152730 | 0.064433 | 0.201232 | 0.901734 | 0.152730 |
| xgboost | 0.189279 | 0.069996 | 0.240252 | 1.000000 | 0.189279 |

## Closer Per Tick

- lstm: `149`
- xgboost: `24`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

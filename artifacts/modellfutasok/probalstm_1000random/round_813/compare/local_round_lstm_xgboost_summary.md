# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m2-train.csv`
- round_num: `3`
- rows: `275`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.183542 | 0.080276 | 0.244333 | 0.883636 | 0.183542 |
| xgboost | 0.267234 | 0.130021 | 0.375388 | 0.665455 | 0.267234 |

## Closer Per Tick

- lstm: `272`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

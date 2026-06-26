# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `22`
- rows: `254`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.545632 | 0.372393 | 1.003528 | 0.444882 | 0.454368 |
| xgboost | 0.426298 | 0.235791 | 0.639175 | 0.444882 | 0.573702 |

## Closer Per Tick

- lstm: `4`
- xgboost: `250`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

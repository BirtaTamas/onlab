# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `16`
- rows: `101`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.464021 | 0.240253 | 0.664570 | 0.564356 | 0.535979 |
| xgboost | 0.406392 | 0.190365 | 0.551038 | 0.841584 | 0.593608 |

## Closer Per Tick

- lstm: `30`
- xgboost: `71`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `15`
- rows: `242`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.412522 | 0.185584 | 0.551126 | 0.822314 | 0.587478 |
| xgboost | 0.317882 | 0.120427 | 0.404149 | 0.933884 | 0.682118 |

## Closer Per Tick

- lstm: `21`
- xgboost: `221`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

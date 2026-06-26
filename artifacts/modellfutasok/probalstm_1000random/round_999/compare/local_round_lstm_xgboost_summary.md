# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `7`
- rows: `140`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.523395 | 0.290135 | 0.777385 | 0.300000 | 0.523395 |
| xgboost | 0.458010 | 0.237677 | 0.684376 | 0.585714 | 0.458010 |

## Closer Per Tick

- lstm: `26`
- xgboost: `114`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

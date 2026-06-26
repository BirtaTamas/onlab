# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `24`
- rows: `216`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.322031 | 0.126123 | 0.410798 | 1.000000 | 0.677969 |
| xgboost | 0.305226 | 0.119002 | 0.388714 | 0.986111 | 0.694774 |

## Closer Per Tick

- lstm: `66`
- xgboost: `150`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

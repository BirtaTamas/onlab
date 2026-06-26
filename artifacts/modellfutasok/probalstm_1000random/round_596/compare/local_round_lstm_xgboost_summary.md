# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `11`
- rows: `198`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.425427 | 0.232389 | 0.779830 | 0.656566 | 0.574573 |
| xgboost | 0.375433 | 0.173276 | 0.544575 | 0.873737 | 0.624567 |

## Closer Per Tick

- lstm: `53`
- xgboost: `145`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

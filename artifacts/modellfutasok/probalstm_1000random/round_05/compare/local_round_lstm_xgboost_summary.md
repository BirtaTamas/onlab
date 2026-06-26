# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `6`
- rows: `219`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.903662 | 0.831503 | 3.072749 | 0.000000 | 0.096338 |
| xgboost | 0.840003 | 0.733502 | 2.263103 | 0.114155 | 0.159997 |

## Closer Per Tick

- lstm: `11`
- xgboost: `208`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

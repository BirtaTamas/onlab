# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `9`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.428429 | 0.197925 | 0.589414 | 0.756522 | 0.571571 |
| xgboost | 0.320808 | 0.111927 | 0.397434 | 1.000000 | 0.679192 |

## Closer Per Tick

- lstm: `0`
- xgboost: `230`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

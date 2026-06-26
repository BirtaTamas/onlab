# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `11`
- rows: `200`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.356203 | 0.162278 | 0.492079 | 0.690000 | 0.643797 |
| xgboost | 0.235723 | 0.083735 | 0.296577 | 0.985000 | 0.764277 |

## Closer Per Tick

- lstm: `2`
- xgboost: `198`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

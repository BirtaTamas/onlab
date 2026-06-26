# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `17`
- rows: `310`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.479689 | 0.290746 | 0.757755 | 0.316129 | 0.479689 |
| xgboost | 0.441555 | 0.260621 | 0.691881 | 0.393548 | 0.441555 |

## Closer Per Tick

- lstm: `81`
- xgboost: `229`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `1`
- rows: `208`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.267505 | 0.108685 | 0.348985 | 0.774038 | 0.732495 |
| xgboost | 0.200570 | 0.083060 | 0.262054 | 1.000000 | 0.799430 |

## Closer Per Tick

- lstm: `7`
- xgboost: `201`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

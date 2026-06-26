# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `10`
- rows: `222`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.396104 | 0.258457 | 0.669986 | 0.468468 | 0.396104 |
| xgboost | 0.504125 | 0.317994 | 0.844610 | 0.387387 | 0.504125 |

## Closer Per Tick

- lstm: `178`
- xgboost: `44`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

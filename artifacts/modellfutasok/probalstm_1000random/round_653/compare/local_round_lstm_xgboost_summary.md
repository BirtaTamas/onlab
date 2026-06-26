# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `15`
- rows: `152`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.005375 | 0.000071 | 0.005410 | 1.000000 | 0.005375 |
| xgboost | 0.021543 | 0.001308 | 0.022228 | 1.000000 | 0.021543 |

## Closer Per Tick

- lstm: `81`
- xgboost: `71`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

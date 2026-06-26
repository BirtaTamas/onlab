# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `22`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.264204 | 0.087596 | 0.322283 | 1.000000 | 0.735796 |
| xgboost | 0.285996 | 0.104966 | 0.358242 | 1.000000 | 0.714004 |

## Closer Per Tick

- lstm: `148`
- xgboost: `82`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

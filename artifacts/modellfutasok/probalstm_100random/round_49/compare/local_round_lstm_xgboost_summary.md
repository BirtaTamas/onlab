# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `8`
- rows: `138`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.380297 | 0.201387 | 0.566902 | 0.710145 | 0.619703 |
| xgboost | 0.377914 | 0.197517 | 0.546674 | 0.543478 | 0.622086 |

## Closer Per Tick

- lstm: `56`
- xgboost: `82`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

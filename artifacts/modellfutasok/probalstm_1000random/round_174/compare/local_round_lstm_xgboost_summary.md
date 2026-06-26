# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `11`
- rows: `212`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.434663 | 0.232423 | 0.666450 | 0.674528 | 0.565337 |
| xgboost | 0.343256 | 0.139910 | 0.445679 | 0.806604 | 0.656744 |

## Closer Per Tick

- lstm: `29`
- xgboost: `183`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

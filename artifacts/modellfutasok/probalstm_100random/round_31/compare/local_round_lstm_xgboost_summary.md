# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `12`
- rows: `122`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.173388 | 0.059715 | 0.216172 | 1.000000 | 0.826612 |
| xgboost | 0.131077 | 0.051630 | 0.168087 | 1.000000 | 0.868923 |

## Closer Per Tick

- lstm: `10`
- xgboost: `112`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

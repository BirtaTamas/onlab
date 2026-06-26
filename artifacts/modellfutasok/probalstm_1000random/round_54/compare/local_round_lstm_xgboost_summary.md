# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `9`
- rows: `138`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.260123 | 0.080859 | 0.312068 | 1.000000 | 0.739877 |
| xgboost | 0.266835 | 0.089477 | 0.325476 | 1.000000 | 0.733165 |

## Closer Per Tick

- lstm: `89`
- xgboost: `49`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

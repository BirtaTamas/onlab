# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `17`
- rows: `141`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.168621 | 0.089373 | 0.243930 | 0.730496 | 0.168621 |
| xgboost | 0.194239 | 0.091249 | 0.269549 | 0.730496 | 0.194239 |

## Closer Per Tick

- lstm: `120`
- xgboost: `21`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

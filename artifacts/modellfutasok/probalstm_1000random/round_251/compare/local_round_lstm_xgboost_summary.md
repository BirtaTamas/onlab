# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `3`
- rows: `153`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.231271 | 0.087452 | 0.294398 | 0.869281 | 0.768729 |
| xgboost | 0.208640 | 0.083061 | 0.270664 | 0.790850 | 0.791360 |

## Closer Per Tick

- lstm: `35`
- xgboost: `118`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

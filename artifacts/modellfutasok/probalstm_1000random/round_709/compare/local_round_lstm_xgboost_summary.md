# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `23`
- rows: `255`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.427691 | 0.206373 | 0.590800 | 0.435294 | 0.427691 |
| xgboost | 0.392392 | 0.166578 | 0.515685 | 0.937255 | 0.392392 |

## Closer Per Tick

- lstm: `57`
- xgboost: `198`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

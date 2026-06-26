# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `19`
- rows: `226`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.354904 | 0.150942 | 0.467385 | 0.818584 | 0.354904 |
| xgboost | 0.398103 | 0.174979 | 0.528351 | 0.818584 | 0.398103 |

## Closer Per Tick

- lstm: `177`
- xgboost: `49`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

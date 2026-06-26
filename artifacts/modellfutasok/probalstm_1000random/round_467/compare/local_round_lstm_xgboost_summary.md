# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `3`
- rows: `204`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.052555 | 0.004947 | 0.055289 | 1.000000 | 0.052555 |
| xgboost | 0.145945 | 0.032007 | 0.165538 | 1.000000 | 0.145945 |

## Closer Per Tick

- lstm: `203`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

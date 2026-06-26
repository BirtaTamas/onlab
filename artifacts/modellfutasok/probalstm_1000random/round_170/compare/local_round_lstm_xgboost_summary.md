# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `16`
- rows: `191`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.078927 | 0.012309 | 0.085928 | 1.000000 | 0.078927 |
| xgboost | 0.155191 | 0.038908 | 0.179187 | 1.000000 | 0.155191 |

## Closer Per Tick

- lstm: `191`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

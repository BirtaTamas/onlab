# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `7`
- rows: `188`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.411167 | 0.189311 | 0.552554 | 0.856383 | 0.411167 |
| xgboost | 0.410800 | 0.184580 | 0.548449 | 0.941489 | 0.410800 |

## Closer Per Tick

- lstm: `103`
- xgboost: `85`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

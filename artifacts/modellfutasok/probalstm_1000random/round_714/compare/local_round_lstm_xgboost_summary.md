# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `8`
- rows: `178`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.719293 | 0.566776 | 1.626724 | 0.078652 | 0.280707 |
| xgboost | 0.610397 | 0.408713 | 1.058802 | 0.398876 | 0.389603 |

## Closer Per Tick

- lstm: `14`
- xgboost: `164`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

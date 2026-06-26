# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `11`
- rows: `142`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.351283 | 0.151723 | 0.472215 | 0.669014 | 0.648717 |
| xgboost | 0.283460 | 0.124301 | 0.385055 | 0.669014 | 0.716540 |

## Closer Per Tick

- lstm: `26`
- xgboost: `116`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

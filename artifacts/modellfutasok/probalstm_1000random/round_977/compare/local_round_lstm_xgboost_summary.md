# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `14`
- rows: `142`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.107409 | 0.012231 | 0.114057 | 1.000000 | 0.892591 |
| xgboost | 0.023966 | 0.000623 | 0.024284 | 1.000000 | 0.976034 |

## Closer Per Tick

- lstm: `0`
- xgboost: `142`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

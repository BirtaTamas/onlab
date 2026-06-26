# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `6`
- rows: `177`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.256297 | 0.083709 | 0.313766 | 1.000000 | 0.743703 |
| xgboost | 0.224131 | 0.088979 | 0.290832 | 0.988701 | 0.775869 |

## Closer Per Tick

- lstm: `64`
- xgboost: `113`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`

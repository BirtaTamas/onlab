# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `4`
- rows: `143`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.489811 | 0.253469 | 0.706870 | 0.643357 | 0.510189 |
| xgboost | 0.598009 | 0.380653 | 0.993670 | 0.027972 | 0.401991 |

## Closer Per Tick

- lstm: `135`
- xgboost: `8`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

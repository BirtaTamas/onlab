# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `17`
- rows: `203`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.169861 | 0.064335 | 0.215871 | 0.965517 | 0.169861 |
| xgboost | 0.190013 | 0.074250 | 0.242813 | 1.000000 | 0.190013 |

## Closer Per Tick

- lstm: `176`
- xgboost: `27`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

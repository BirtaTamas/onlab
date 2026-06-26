# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `5`
- rows: `307`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.460814 | 0.279921 | 0.721050 | 0.322476 | 0.460814 |
| xgboost | 0.369154 | 0.173110 | 0.501532 | 0.723127 | 0.369154 |

## Closer Per Tick

- lstm: `71`
- xgboost: `236`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

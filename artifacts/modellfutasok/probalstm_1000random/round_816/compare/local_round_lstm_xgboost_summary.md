# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `3`
- rows: `124`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.169669 | 0.037119 | 0.191985 | 1.000000 | 0.830331 |
| xgboost | 0.181706 | 0.047960 | 0.211755 | 1.000000 | 0.818294 |

## Closer Per Tick

- lstm: `66`
- xgboost: `58`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

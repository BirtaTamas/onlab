# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `14`
- rows: `227`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.159360 | 0.041619 | 0.185264 | 1.000000 | 0.159360 |
| xgboost | 0.181694 | 0.055371 | 0.217784 | 1.000000 | 0.181694 |

## Closer Per Tick

- lstm: `170`
- xgboost: `57`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

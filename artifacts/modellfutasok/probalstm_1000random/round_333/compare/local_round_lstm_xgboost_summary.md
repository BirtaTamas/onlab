# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `16`
- rows: `198`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.270613 | 0.128124 | 0.368260 | 0.767677 | 0.270613 |
| xgboost | 0.312206 | 0.163218 | 0.445352 | 0.484848 | 0.312206 |

## Closer Per Tick

- lstm: `196`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

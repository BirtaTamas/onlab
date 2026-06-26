# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `9`
- rows: `208`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.730479 | 0.544742 | 1.389123 | 0.038462 | 0.730479 |
| xgboost | 0.743153 | 0.566318 | 1.482456 | 0.000000 | 0.743153 |

## Closer Per Tick

- lstm: `123`
- xgboost: `85`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

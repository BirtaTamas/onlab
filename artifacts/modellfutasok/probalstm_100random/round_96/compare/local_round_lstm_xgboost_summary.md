# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `13`
- rows: `139`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.515650 | 0.269450 | 0.732957 | 0.172662 | 0.515650 |
| xgboost | 0.550133 | 0.316097 | 0.849203 | 0.143885 | 0.550133 |

## Closer Per Tick

- lstm: `54`
- xgboost: `85`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

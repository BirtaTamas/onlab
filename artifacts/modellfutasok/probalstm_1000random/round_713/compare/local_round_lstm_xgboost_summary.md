# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `8`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.280291 | 0.115516 | 0.364357 | 0.969565 | 0.719709 |
| xgboost | 0.237368 | 0.089967 | 0.300837 | 1.000000 | 0.762632 |

## Closer Per Tick

- lstm: `26`
- xgboost: `204`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `10`
- rows: `156`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.285213 | 0.098585 | 0.351675 | 1.000000 | 0.714787 |
| xgboost | 0.238260 | 0.077267 | 0.290885 | 0.891026 | 0.761740 |

## Closer Per Tick

- lstm: `23`
- xgboost: `133`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

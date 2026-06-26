# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `1`
- rows: `169`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.299462 | 0.130630 | 0.402135 | 0.698225 | 0.700538 |
| xgboost | 0.241973 | 0.109904 | 0.327685 | 0.615385 | 0.758027 |

## Closer Per Tick

- lstm: `26`
- xgboost: `143`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

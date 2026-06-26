# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `5`
- rows: `246`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.342355 | 0.127729 | 0.429193 | 1.000000 | 0.657645 |
| xgboost | 0.312289 | 0.107519 | 0.383329 | 1.000000 | 0.687711 |

## Closer Per Tick

- lstm: `31`
- xgboost: `215`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

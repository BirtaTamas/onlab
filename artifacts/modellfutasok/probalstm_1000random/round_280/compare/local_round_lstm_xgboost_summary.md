# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `3`
- rows: `105`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.048652 | 0.003091 | 0.050280 | 1.000000 | 0.951348 |
| xgboost | 0.015461 | 0.000371 | 0.015650 | 1.000000 | 0.984539 |

## Closer Per Tick

- lstm: `0`
- xgboost: `105`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

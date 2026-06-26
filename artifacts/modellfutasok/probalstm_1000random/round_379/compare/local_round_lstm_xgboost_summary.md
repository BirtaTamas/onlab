# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `19`
- rows: `115`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.254299 | 0.098612 | 0.333407 | 0.791304 | 0.745701 |
| xgboost | 0.180052 | 0.054116 | 0.217595 | 1.000000 | 0.819948 |

## Closer Per Tick

- lstm: `1`
- xgboost: `114`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `6`
- rows: `190`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.151532 | 0.054462 | 0.189619 | 1.000000 | 0.151532 |
| xgboost | 0.186206 | 0.075948 | 0.241723 | 0.973684 | 0.186206 |

## Closer Per Tick

- lstm: `180`
- xgboost: `10`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

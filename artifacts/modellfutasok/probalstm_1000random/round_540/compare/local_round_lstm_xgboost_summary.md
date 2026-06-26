# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `10`
- rows: `262`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.319069 | 0.124031 | 0.412215 | 0.927481 | 0.680931 |
| xgboost | 0.298479 | 0.129879 | 0.423381 | 0.912214 | 0.701521 |

## Closer Per Tick

- lstm: `92`
- xgboost: `170`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

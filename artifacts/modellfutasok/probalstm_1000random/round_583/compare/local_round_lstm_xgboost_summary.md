# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-astralis-vs-wildcard-bo3-qSXX__H_dx2QMbEuGWf0Qb/astralis-vs-wildcard-m2-mirage.csv`
- round_num: `15`
- rows: `196`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.204430 | 0.049997 | 0.235458 | 1.000000 | 0.795570 |
| xgboost | 0.255717 | 0.077742 | 0.306575 | 1.000000 | 0.744283 |

## Closer Per Tick

- lstm: `162`
- xgboost: `34`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `12`
- rows: `190`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.308080 | 0.109072 | 0.381456 | 1.000000 | 0.691920 |
| xgboost | 0.265804 | 0.085804 | 0.321672 | 1.000000 | 0.734196 |

## Closer Per Tick

- lstm: `21`
- xgboost: `169`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

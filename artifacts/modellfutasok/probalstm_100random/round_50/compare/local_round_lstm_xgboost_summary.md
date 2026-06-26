# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `16`
- rows: `111`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.386601 | 0.177043 | 0.520819 | 0.693694 | 0.613399 |
| xgboost | 0.335438 | 0.136514 | 0.433315 | 1.000000 | 0.664562 |

## Closer Per Tick

- lstm: `4`
- xgboost: `107`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

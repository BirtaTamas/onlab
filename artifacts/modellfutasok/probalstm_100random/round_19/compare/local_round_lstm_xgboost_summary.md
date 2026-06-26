# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `5`
- rows: `152`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.190625 | 0.104696 | 0.280576 | 0.710526 | 0.190625 |
| xgboost | 0.198920 | 0.105767 | 0.290097 | 0.710526 | 0.198920 |

## Closer Per Tick

- lstm: `128`
- xgboost: `24`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

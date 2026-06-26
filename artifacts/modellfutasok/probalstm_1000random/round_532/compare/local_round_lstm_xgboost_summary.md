# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `4`
- rows: `194`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.136726 | 0.048387 | 0.169659 | 1.000000 | 0.136726 |
| xgboost | 0.178920 | 0.069898 | 0.228651 | 1.000000 | 0.178920 |

## Closer Per Tick

- lstm: `194`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

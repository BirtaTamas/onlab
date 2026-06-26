# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `14`
- rows: `159`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.459460 | 0.299887 | 0.993868 | 0.672956 | 0.459460 |
| xgboost | 0.503989 | 0.332735 | 1.116342 | 0.672956 | 0.503989 |

## Closer Per Tick

- lstm: `144`
- xgboost: `15`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

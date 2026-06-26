# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-spirit-bo3-NmwBJVzYbgyZgcQrbNESHr/flyquest-vs-spirit-m1-anubis.csv`
- round_num: `14`
- rows: `130`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.022692 | 0.001418 | 0.023466 | 1.000000 | 0.022692 |
| xgboost | 0.029995 | 0.001833 | 0.030982 | 1.000000 | 0.029995 |

## Closer Per Tick

- lstm: `108`
- xgboost: `22`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

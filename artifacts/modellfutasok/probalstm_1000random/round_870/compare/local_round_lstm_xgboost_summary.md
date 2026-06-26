# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-spirit-bo3-NmwBJVzYbgyZgcQrbNESHr/flyquest-vs-spirit-m1-anubis.csv`
- round_num: `10`
- rows: `309`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.397172 | 0.198976 | 0.656665 | 0.825243 | 0.397172 |
| xgboost | 0.391788 | 0.195873 | 0.670511 | 0.805825 | 0.391788 |

## Closer Per Tick

- lstm: `110`
- xgboost: `199`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`

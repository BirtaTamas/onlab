# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `23`
- rows: `106`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.493118 | 0.258655 | 0.713759 | 0.698113 | 0.506882 |
| xgboost | 0.488199 | 0.267855 | 0.730503 | 0.584906 | 0.511801 |

## Closer Per Tick

- lstm: `40`
- xgboost: `66`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

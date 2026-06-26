# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `13`
- rows: `114`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.356650 | 0.161539 | 0.478970 | 0.885965 | 0.643350 |
| xgboost | 0.327003 | 0.151781 | 0.442255 | 0.947368 | 0.672997 |

## Closer Per Tick

- lstm: `30`
- xgboost: `84`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

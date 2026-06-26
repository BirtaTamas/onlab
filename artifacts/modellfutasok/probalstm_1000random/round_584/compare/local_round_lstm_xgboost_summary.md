# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `5`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.198819 | 0.069624 | 0.247381 | 0.995652 | 0.801181 |
| xgboost | 0.263395 | 0.113332 | 0.351900 | 0.604348 | 0.736605 |

## Closer Per Tick

- lstm: `229`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

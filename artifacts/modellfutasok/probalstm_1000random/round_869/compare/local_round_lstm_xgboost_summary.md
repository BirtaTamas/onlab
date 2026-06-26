# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `24`
- rows: `243`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.148039 | 0.028788 | 0.164749 | 1.000000 | 0.148039 |
| xgboost | 0.241845 | 0.070212 | 0.286156 | 1.000000 | 0.241845 |

## Closer Per Tick

- lstm: `243`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

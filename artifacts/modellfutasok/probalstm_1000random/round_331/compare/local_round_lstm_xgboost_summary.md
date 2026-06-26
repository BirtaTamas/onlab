# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `11`
- rows: `138`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.342390 | 0.177210 | 0.484352 | 0.536232 | 0.342390 |
| xgboost | 0.349582 | 0.181494 | 0.497429 | 0.442029 | 0.349582 |

## Closer Per Tick

- lstm: `94`
- xgboost: `44`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

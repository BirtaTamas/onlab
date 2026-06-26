# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `4`
- rows: `152`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.194992 | 0.077726 | 0.252678 | 0.907895 | 0.194992 |
| xgboost | 0.250219 | 0.106524 | 0.335684 | 0.743421 | 0.250219 |

## Closer Per Tick

- lstm: `127`
- xgboost: `25`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

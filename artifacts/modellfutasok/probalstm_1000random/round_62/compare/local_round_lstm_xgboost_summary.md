# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `14`
- rows: `246`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.369336 | 0.152158 | 0.478530 | 0.971545 | 0.630664 |
| xgboost | 0.338782 | 0.133851 | 0.433837 | 0.975610 | 0.661218 |

## Closer Per Tick

- lstm: `70`
- xgboost: `176`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

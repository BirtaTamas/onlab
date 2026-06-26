# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `5`
- rows: `181`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.139888 | 0.030296 | 0.158105 | 1.000000 | 0.860112 |
| xgboost | 0.148976 | 0.039649 | 0.173784 | 1.000000 | 0.851024 |

## Closer Per Tick

- lstm: `85`
- xgboost: `96`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

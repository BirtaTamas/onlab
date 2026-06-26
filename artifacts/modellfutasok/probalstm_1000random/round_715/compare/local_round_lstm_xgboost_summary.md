# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `14`
- rows: `229`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.472667 | 0.234539 | 0.654793 | 0.471616 | 0.472667 |
| xgboost | 0.476263 | 0.237958 | 0.666644 | 0.886463 | 0.476263 |

## Closer Per Tick

- lstm: `82`
- xgboost: `147`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

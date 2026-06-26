# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `11`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.133670 | 0.034203 | 0.155626 | 1.000000 | 0.866330 |
| xgboost | 0.144916 | 0.043894 | 0.175341 | 1.000000 | 0.855084 |

## Closer Per Tick

- lstm: `148`
- xgboost: `82`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `10`
- rows: `191`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.533083 | 0.313736 | 0.833994 | 0.465969 | 0.466917 |
| xgboost | 0.565072 | 0.351637 | 0.913692 | 0.345550 | 0.434928 |

## Closer Per Tick

- lstm: `155`
- xgboost: `36`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `2`
- rows: `107`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.329835 | 0.140247 | 0.433251 | 0.943925 | 0.670165 |
| xgboost | 0.299783 | 0.133859 | 0.400802 | 1.000000 | 0.700217 |

## Closer Per Tick

- lstm: `50`
- xgboost: `57`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

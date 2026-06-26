# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `6`
- rows: `170`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.096688 | 0.026322 | 0.114758 | 1.000000 | 0.903312 |
| xgboost | 0.114062 | 0.035418 | 0.139761 | 1.000000 | 0.885938 |

## Closer Per Tick

- lstm: `131`
- xgboost: `39`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

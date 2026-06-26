# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `10`
- rows: `166`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.290116 | 0.102475 | 0.364000 | 0.939759 | 0.709884 |
| xgboost | 0.250209 | 0.092196 | 0.320141 | 0.939759 | 0.749791 |

## Closer Per Tick

- lstm: `47`
- xgboost: `119`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m2-inferno.csv`
- round_num: `17`
- rows: `180`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.355828 | 0.197315 | 0.529425 | 0.527778 | 0.355828 |
| xgboost | 0.332458 | 0.167411 | 0.472729 | 0.555556 | 0.332458 |

## Closer Per Tick

- lstm: `60`
- xgboost: `120`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

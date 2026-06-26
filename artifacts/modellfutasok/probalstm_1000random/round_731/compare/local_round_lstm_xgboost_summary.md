# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `17`
- rows: `191`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.161152 | 0.029732 | 0.178277 | 1.000000 | 0.838848 |
| xgboost | 0.128661 | 0.019074 | 0.139287 | 1.000000 | 0.871339 |

## Closer Per Tick

- lstm: `8`
- xgboost: `183`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

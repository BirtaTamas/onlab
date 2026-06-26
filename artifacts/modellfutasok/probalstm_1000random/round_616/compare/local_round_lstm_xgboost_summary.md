# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `10`
- rows: `221`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.372543 | 0.158502 | 0.491470 | 0.941176 | 0.627457 |
| xgboost | 0.347854 | 0.139677 | 0.453995 | 0.941176 | 0.652146 |

## Closer Per Tick

- lstm: `57`
- xgboost: `164`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `15`
- rows: `248`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.270061 | 0.143672 | 0.391272 | 0.564516 | 0.270061 |
| xgboost | 0.258713 | 0.121636 | 0.356468 | 0.657258 | 0.258713 |

## Closer Per Tick

- lstm: `135`
- xgboost: `113`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

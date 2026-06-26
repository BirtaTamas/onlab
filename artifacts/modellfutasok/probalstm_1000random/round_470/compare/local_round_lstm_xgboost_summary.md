# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `14`
- rows: `184`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.062852 | 0.004234 | 0.065076 | 1.000000 | 0.937148 |
| xgboost | 0.015076 | 0.000241 | 0.015198 | 1.000000 | 0.984924 |

## Closer Per Tick

- lstm: `0`
- xgboost: `184`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

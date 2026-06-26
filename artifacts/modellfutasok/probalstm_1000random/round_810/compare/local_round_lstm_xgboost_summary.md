# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `7`
- rows: `222`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.500765 | 0.280497 | 0.816218 | 0.418919 | 0.499235 |
| xgboost | 0.449026 | 0.228985 | 0.662132 | 0.545045 | 0.550974 |

## Closer Per Tick

- lstm: `66`
- xgboost: `156`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

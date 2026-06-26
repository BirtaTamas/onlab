# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `7`
- rows: `167`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.479683 | 0.262041 | 0.717289 | 0.586826 | 0.520317 |
| xgboost | 0.466418 | 0.247862 | 0.678503 | 0.371257 | 0.533582 |

## Closer Per Tick

- lstm: `82`
- xgboost: `85`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

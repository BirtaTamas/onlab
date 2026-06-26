# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `4`
- rows: `260`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.590066 | 0.376679 | 0.982593 | 0.303846 | 0.590066 |
| xgboost | 0.582116 | 0.360073 | 0.931600 | 0.223077 | 0.582116 |

## Closer Per Tick

- lstm: `112`
- xgboost: `148`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

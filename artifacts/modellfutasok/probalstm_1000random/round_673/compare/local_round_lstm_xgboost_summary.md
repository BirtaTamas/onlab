# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `8`
- rows: `101`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.428379 | 0.225000 | 0.621672 | 0.514851 | 0.428379 |
| xgboost | 0.627271 | 0.423991 | 1.129416 | 0.257426 | 0.627271 |

## Closer Per Tick

- lstm: `94`
- xgboost: `7`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

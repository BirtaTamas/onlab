# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `18`
- rows: `147`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.357620 | 0.135734 | 0.450396 | 1.000000 | 0.642380 |
| xgboost | 0.384089 | 0.157109 | 0.494374 | 1.000000 | 0.615911 |

## Closer Per Tick

- lstm: `117`
- xgboost: `30`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `6`
- rows: `100`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.017510 | 0.000411 | 0.017719 | 1.000000 | 0.017510 |
| xgboost | 0.026458 | 0.000793 | 0.026863 | 1.000000 | 0.026458 |

## Closer Per Tick

- lstm: `90`
- xgboost: `10`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

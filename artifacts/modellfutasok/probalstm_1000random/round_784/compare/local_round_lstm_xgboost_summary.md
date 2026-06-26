# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `1`
- rows: `188`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.372361 | 0.157099 | 0.487432 | 0.845745 | 0.627639 |
| xgboost | 0.248230 | 0.094505 | 0.318338 | 0.808511 | 0.751770 |

## Closer Per Tick

- lstm: `41`
- xgboost: `147`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

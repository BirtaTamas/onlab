# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m2-dust2.csv`
- round_num: `8`
- rows: `157`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.441494 | 0.212450 | 0.606830 | 0.439490 | 0.558506 |
| xgboost | 0.391818 | 0.173409 | 0.520908 | 1.000000 | 0.608182 |

## Closer Per Tick

- lstm: `2`
- xgboost: `155`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `13`
- rows: `110`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.378651 | 0.158060 | 0.494459 | 0.945455 | 0.621349 |
| xgboost | 0.233405 | 0.076398 | 0.286637 | 1.000000 | 0.766595 |

## Closer Per Tick

- lstm: `0`
- xgboost: `110`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

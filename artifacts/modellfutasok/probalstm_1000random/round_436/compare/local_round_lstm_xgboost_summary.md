# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `12`
- rows: `156`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.337988 | 0.132600 | 0.430243 | 1.000000 | 0.662012 |
| xgboost | 0.364743 | 0.153346 | 0.474691 | 1.000000 | 0.635257 |

## Closer Per Tick

- lstm: `122`
- xgboost: `34`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `7`
- rows: `268`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.372960 | 0.177780 | 0.509920 | 0.679104 | 0.372960 |
| xgboost | 0.401414 | 0.191086 | 0.548042 | 0.835821 | 0.401414 |

## Closer Per Tick

- lstm: `172`
- xgboost: `96`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `15`
- rows: `224`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.091510 | 0.012230 | 0.098411 | 1.000000 | 0.908490 |
| xgboost | 0.044104 | 0.002735 | 0.045544 | 1.000000 | 0.955896 |

## Closer Per Tick

- lstm: `0`
- xgboost: `224`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

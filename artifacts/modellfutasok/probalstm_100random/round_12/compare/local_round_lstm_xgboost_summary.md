# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `8`
- rows: `225`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.259387 | 0.093586 | 0.327032 | 0.911111 | 0.740613 |
| xgboost | 0.215265 | 0.074334 | 0.268772 | 1.000000 | 0.784735 |

## Closer Per Tick

- lstm: `10`
- xgboost: `215`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

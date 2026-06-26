# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-lynn-vision-bo3-0ZNMTlQ0ZfadRgwA0Ax5fN/3dmax-vs-lynn-vision-m2-anubis.csv`
- round_num: `14`
- rows: `204`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.385247 | 0.174552 | 0.520359 | 0.803922 | 0.614753 |
| xgboost | 0.302872 | 0.115841 | 0.385811 | 0.833333 | 0.697128 |

## Closer Per Tick

- lstm: `30`
- xgboost: `174`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

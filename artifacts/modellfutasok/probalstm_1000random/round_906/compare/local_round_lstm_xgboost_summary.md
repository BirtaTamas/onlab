# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `4`
- rows: `129`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.151931 | 0.059440 | 0.197600 | 0.844961 | 0.848069 |
| xgboost | 0.107804 | 0.041897 | 0.138203 | 1.000000 | 0.892196 |

## Closer Per Tick

- lstm: `0`
- xgboost: `129`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

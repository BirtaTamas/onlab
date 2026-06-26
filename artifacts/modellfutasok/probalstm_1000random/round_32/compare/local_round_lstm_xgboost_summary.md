# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `18`
- rows: `158`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.312077 | 0.178849 | 0.538172 | 0.740506 | 0.687923 |
| xgboost | 0.280741 | 0.168403 | 0.464774 | 0.740506 | 0.719259 |

## Closer Per Tick

- lstm: `45`
- xgboost: `113`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

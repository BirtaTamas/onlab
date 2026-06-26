# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `14`
- rows: `125`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.360422 | 0.169437 | 0.504731 | 0.712000 | 0.360422 |
| xgboost | 0.395246 | 0.198356 | 0.598531 | 0.704000 | 0.395246 |

## Closer Per Tick

- lstm: `91`
- xgboost: `34`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

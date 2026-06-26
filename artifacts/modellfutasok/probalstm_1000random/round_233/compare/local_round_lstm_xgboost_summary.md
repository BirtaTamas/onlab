# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `13`
- rows: `102`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.489515 | 0.251940 | 0.691763 | 0.490196 | 0.510485 |
| xgboost | 0.458694 | 0.225790 | 0.633892 | 0.372549 | 0.541306 |

## Closer Per Tick

- lstm: `26`
- xgboost: `76`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`

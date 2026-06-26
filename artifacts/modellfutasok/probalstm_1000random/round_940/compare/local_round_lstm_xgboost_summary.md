# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `10`
- rows: `306`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.281264 | 0.139688 | 0.392535 | 0.588235 | 0.281264 |
| xgboost | 0.311510 | 0.162260 | 0.446694 | 0.529412 | 0.311510 |

## Closer Per Tick

- lstm: `270`
- xgboost: `36`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

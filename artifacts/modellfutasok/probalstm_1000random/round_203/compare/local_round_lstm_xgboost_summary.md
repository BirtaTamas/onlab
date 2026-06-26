# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `8`
- rows: `244`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.414679 | 0.190205 | 0.595675 | 0.901639 | 0.585321 |
| xgboost | 0.441903 | 0.211946 | 0.628040 | 0.918033 | 0.558097 |

## Closer Per Tick

- lstm: `206`
- xgboost: `38`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

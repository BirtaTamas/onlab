# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `11`
- rows: `200`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.662219 | 0.453336 | 1.193619 | 0.010000 | 0.662219 |
| xgboost | 0.681354 | 0.483463 | 1.367014 | 0.020000 | 0.681354 |

## Closer Per Tick

- lstm: `141`
- xgboost: `59`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`

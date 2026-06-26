# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `7`
- rows: `116`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.112209 | 0.025408 | 0.127793 | 1.000000 | 0.112209 |
| xgboost | 0.240151 | 0.074045 | 0.288324 | 1.000000 | 0.240151 |

## Closer Per Tick

- lstm: `116`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
